import pennylane as qml

import matplotlib.pyplot as plt
import math
from pennylane.optimize import AdamOptimizer
import sys
import time
import numpy as np
from pennylane.optimize import AdamOptimizer
from pennylane import numpy as qml_np
import pennylane as qml

import math
import matplotlib.pyplot as plt
import datetime

#======================================
seed=42
# seed = int(time.time())
np.random.seed(seed)
print("Seed:", seed)
num_qubits = 15
n_layers = 2
#===================================================
import subprocess
import tracemalloc

# GPU mem usage
def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], capture_output=True)
    output = result.stdout.decode('utf-8').strip()
    memory_usages = output.split('\n')
    total_memory_usage = sum(map(int, memory_usages))
    return total_memory_usage
# CPU mem Usage
tracemalloc.start()

#===================================================
def random_oracle_builder2(num_qubits):
    size = 2**num_qubits
    negative_index = int(size/2)
    rand_vec = np.zeros(size)
    classical_min_index = size+1
    # Generate indices for the zero elements
    # num_ones = np.random.randint(int(size/(2**num_qubits-2)),int(size/8))
    num_ones = np.random.randint(1,int(size/100)+2)
    one_indices = np.sort(np.random.choice(size, num_ones, replace=False))
    # Set the elements at the zero indices to zero
    min_index_finded = False
    for i in one_indices:
        if not min_index_finded:
            #for positive
            if i<negative_index and i<classical_min_index:
                classical_min_index=i
            #for negative
            # else:
            elif i>=negative_index: 
                classical_min_index = i
                temp = np.binary_repr(classical_min_index,num_qubits)
                temp=int(temp,2)-(1<<num_qubits) 
                min_index_finded = True

        rand_vec[i]=np.random.rand()
    rand_vec = rand_vec/np.linalg.norm(rand_vec)
    
    G_m = None
    if classical_min_index<negative_index:
        G_m = classical_min_index
    else:
        bin_rep_G_m = np.binary_repr(classical_min_index,num_qubits)
        G_m=int(bin_rep_G_m,2)-(1<<num_qubits)
    
    return rand_vec, G_m
# rand_vec,G_m = random_oracle_builder2(20)
# print(G_m)
#===================================================

test_input,G_m = random_oracle_builder2(num_qubits=num_qubits)
print('Classical G_m = ',G_m)

num_overflow_bit = 2
num_qubits = num_qubits + num_overflow_bit
# print('num_overflow_bit = ',num_overflow_bit)
print('num_qubits = ',num_qubits)
print('num_qubits_vqs = ',num_qubits+1)
print("n_layers:", n_layers)

start_state = test_input
# print(start_state)
#===================================================

def add_k_sign(k, wires):
    #sign handling
    bin_rep = np.binary_repr(k,len(wires))
    k = int(bin_rep,2)

    qml.QFT(wires=wires)
    for j in range(len(wires)):
        qml.RZ(k * np.pi / (2**j), wires=wires[j])
    qml.adjoint(qml.QFT)(wires=wires)
#===================================================



#device_name = 'lightning.kokkos'  
#device_name2 = 'lightning.kokkos'

# device_name = 'lightning.qubit'  
# device_name2 = 'lightning.qubit'  
# device_name = 'lightning.qubit'  
# device_name2 = 'lightning.qubit'  
# device_name = 'lightning.gpu'  
# device_name2 = 'lightning.gpu'  
device_name = 'default.qubit' 
device_name2 = 'default.qubit'  
# device_name = 'qulacs.simulator' 
# device_name2 = 'qulacs.simulator' 

print('device name = ',device_name)

#===================================================

import tracemalloc

def vqs(oracle_state,shift,threshold,n_layers,n_shots=None): #True, remainings

   # print('run VQS with parameters: shift=',shift,' threshold=',threshold,' n_layers=',n_layers,' n_shots=',n_shots)
    print('run VQS with parameters: shift=',shift,' threshold=',threshold,' n_layers=',n_layers)
    val_global = []
    
    start_state=oracle_state
    
    
    #returns 
    remainings=[]
    
    
    # vqs
    num_qubit_vqs = 1+num_qubits
    eps_val_q = 1/math.sqrt(2**num_qubit_vqs)/100
    eps_val = min(1e-10, eps_val_q)
    tiny_change_threshold = 1e-4
    cnt_threshold_no_change = 5
    
    
    #helpers
    hsize = 2**num_qubits
    negative_index = int(hsize/2)
    
    


    N = 2**(num_qubit_vqs-2)
    normal_val = math.sqrt(1/N)
    initial_state2 = start_state.tolist()
    initial_state_phi1 = initial_state2
    # print(f'initial_state_phi1={initial_state_phi1}')
    



    def oracle_builder_for_HT_HTZ():
        qml.QubitStateVector(np.array(start_state), wires=range(
            1+num_overflow_bit, num_qubit_vqs))
        
        # add some bits for handle overflow
        for w in reversed(range(2, num_overflow_bit+2)):
            qml.CNOT([w, w-1])
        add_k_sign(shift, wires=range(1, num_qubit_vqs))

    def oracle_builder_for_no_HT_HTZ():
        qml.QubitStateVector(np.array(start_state), wires=range(
        num_overflow_bit, num_qubit_vqs-1))
        # add some bits for handle overflow
        for w in reversed(range(1, num_overflow_bit+1)):
            qml.CNOT([w, w-1])
        add_k_sign(shift, wires=range(0, num_qubit_vqs-1))
        

    def layer_t3_no_HT(theta, qubit_posi):
        # type-2 layer
        # length of theta: (num_qubit_vqs-1)*2
        # length of qubit_posi: num_qubit_vqs-1
        # number of wires: num_qubit_vqs
        for i in range(num_qubit_vqs-1):
            qml.RY(theta[i], wires=(qubit_posi[i]))
        for i in np.arange(0, num_qubit_vqs-2, 2):
            #         qml.ctrl(qml.PauliZ(qubit_posi[i+1]), qubit_posi[i]) # CZ struct2
            qml.CNOT(wires=(qubit_posi[i], qubit_posi[i+1]))  # CNOT struct3
        for i in range(num_qubit_vqs-1):
            qml.RY(theta[i+num_qubit_vqs-1], wires=(qubit_posi[i]))
        for i in np.arange(1, num_qubit_vqs-2, 2):
            #         qml.ctrl(qml.PauliZ(qubit_posi[i+1]), qubit_posi[i]) # CZ struct2
            qml.CNOT(wires=(qubit_posi[i], qubit_posi[i+1]))  # CNOT struct3
    #     qml.ctrl(qml.PauliZ(qubit_posi[0]), qubit_posi[-1]) # CZ struct2
        qml.CNOT(wires=(qubit_posi[-1], qubit_posi[0]))  # CNOT struct3


    def layer_t3_with_HT(theta, num_qubit_vqs):
        # type-2 layer
        # length of theta: (num_qubit_vqs-1)*2
        # number of wires: num_qubit_vqs
        for i in range(num_qubit_vqs-1):
            qml.CRY(theta[i], wires=(0, i+1))
        for i in np.arange(0, num_qubit_vqs-2, 2):
            #         qml.ctrl(qml.PauliZ(i+2), (0, i+1)) # CZ struct2
            qml.Toffoli(wires=(0, i+1, i+2))  # CCNOT struct3

        for i in range(num_qubit_vqs-1):
            qml.CRY(theta[i+num_qubit_vqs-1], wires=(0, i+1))
        for i in np.arange(1, num_qubit_vqs-2, 2):
            #         qml.ctrl(qml.PauliZ(i+2), (0, i+1)) # CZ struct2
            qml.Toffoli(wires=(0, i+1, i+2))  # CCNOT struct3
    #     qml.ctrl(qml.PauliZ(1), (0, num_qubit_vqs-1)) # CZ struct2
        qml.Toffoli(wires=(0, num_qubit_vqs-1, 1))  # CCNOT struct3





    # dev_with_HT=qml.device(device_name2, wires=num_qubit_vqs+1) #AerDevice(wires=num_qubit_vqs, shots=20000, backend='qasm_simulator')
    dev_with_HT = qml.device(device_name, wires=num_qubit_vqs,shots=n_shots)


    @qml.qnode(dev_with_HT)
    def quantum_circuit_with_HT(theta):
        oracle_builder_for_HT_HTZ()
        qml.Hadamard(0)
        for theta_i in theta:
            layer_t3_with_HT(theta_i, num_qubit_vqs)
        qml.Hadamard(0)
        
        return qml.expval(qml.PauliZ(0))

    # dev_with_HTZ=qml.device(device_name2, wires=num_qubit_vqs+1) #AerDevice(wires=num_qubit_vqs, shots=20000, backend='qasm_simulator')
    dev_with_HTZ = qml.device(device_name, wires=num_qubit_vqs,shots=n_shots)


    @qml.qnode(dev_with_HTZ)
    def quantum_circuit_with_HTZ(theta):
        oracle_builder_for_HT_HTZ()
        qml.Hadamard(0)
        for theta_i in theta:
            layer_t3_with_HT(theta_i, num_qubit_vqs)
        qml.CZ([0, 1])
        qml.Hadamard(0)
        return qml.expval(qml.PauliZ(0))


    dev_no_HT_S = qml.device(device_name2, wires=num_qubit_vqs-1,shots=n_shots)
    @qml.qnode(dev_no_HT_S)
    def quantum_circuit_no_HT_return_state(theta):

        oracle_builder_for_no_HT_HTZ()

        for theta_i in theta:
            layer_t3_no_HT(theta_i, list(range(num_qubit_vqs-1)))
        return qml.state()



    


    def objective_fn(theta):
        # global val_global
        val1_1 = quantum_circuit_with_HT(theta)
        val1_2 = quantum_circuit_with_HTZ(theta)
        val1_1 = val1_1/normal_val
        val1_2 = val1_2/normal_val
        obj = -0.5*(val1_1 - val1_2)
        val_global.append(
            [val1_1._value.tolist(), val1_2._value.tolist(), obj._value.tolist()])
        
        return obj


    iter_max = 300  # 300
    num_of_layers = n_layers
    obj_list_rep = []
    theta_list = []
    iter_terminate_list = []




    print_flag = True


    # start_time = datetime.datetime.now()


    optimizer = AdamOptimizer(0.05, beta1=0.9, beta2=0.999)
    theta = qml_np.random.uniform(
        0, 2*math.pi, size=(num_of_layers, 2*(num_qubit_vqs-1)), requires_grad=True)
    obj_list = []
    tiny_change_cnt = 0
    break_flag = False
    iter_terminate = iter_max
    for iter in range(1, iter_max+1):
        theta, obj = optimizer.step_and_cost(objective_fn, theta)
        val1_1 = val_global[-1][0]
        val1_2 = val_global[-1][1]
        if iter >= 2:
            val1_1_old = val_global[-2][0]
            val1_2_old = val_global[-2][1]
        else:
            val1_1_old = 999
            val1_2_old = 999
        val1 = val1_1 - val1_2
        val1_old = val1_1_old - val1_2_old
        if abs(val1) > eps_val:  # eps_val=1e-10
            if abs((val1-val1_old)/val1) < tiny_change_threshold:  # 1e-3
                tiny_change_cnt += 1
            else:
                tiny_change_cnt = 0
        if tiny_change_cnt >= cnt_threshold_no_change:  # no change for a consequtive of 5 iterations, then break
            break_flag = True
        if (iter == 1 or iter % 30 == 0 or iter == iter_max) and print_flag:
            #TODO ? 
            if (np.isclose(val1_1-val1_2,0)):
                return False
            # print(f'iter={iter:3d} :: obj={obj:12.8f} :: val1_1={val1_1:12.8f} :: val1_2={val1_2:12.8f} :: -0.5*(val1_1 - val1_2)={-0.5*(val1_1 - val1_2):12.8f}')

        obj_list.append(obj)
        if break_flag:
            iter_terminate = iter
            break
    theta_list.append(theta)
    obj_list_rep.append(obj_list)

    val_global = []  # reset to empty

    # display the amplified state
    state = quantum_circuit_no_HT_return_state(theta)
    prb = [i.item()**2 for i in state]
    iter_terminate_list.append(iter_terminate)


    # M_r = None
    # min_index = hsize+1
    # min_finded = False
    number_of_negatives = 0


    # return True,remainings
    return np.linalg.norm(prb[-1])
                    
#===================================================
start_time = datetime.datetime.now()

Low = -2**(num_qubits-num_overflow_bit-1)
High = 2**(num_qubits-num_overflow_bit-1)+1

previous_s=None
S=None
counter_steps=0
while True:
    counter_steps+=1
    if previous_s==S and previous_s!=None:
        QG_m = -1-previous_s
        print('QG_m = ',-1-previous_s)
        break 

    previous_s=S
    S = math.floor((Low+High)/2)
    
    results_minus_1=vqs(start_state,S,threshold=1,n_layers=n_layers,n_shots=None)
    if results_minus_1==False:
        High = S
    elif results_minus_1<0.95:
        Low = S
    else:
        QG_m = -1-S
        print('QG_m = ',-1-S)
        break 
          

print('steps: ',counter_steps)
    
end_time = datetime.datetime.now()
duration = end_time - start_time
duration_in_s = duration.total_seconds()
print(f'time consumed: {duration_in_s}s')
print('CPU Memory usage :',tracemalloc.get_traced_memory())
tracemalloc.stop()
gpu_memory_usage = get_gpu_memory_usage()
print(f"GPU memory usage: {gpu_memory_usage} MiB")        
print('done')


#===================================================