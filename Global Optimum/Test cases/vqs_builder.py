# !pip install pennylane
# !pip install pennylane-qiskit
# !pip install pennylane-lightning
# !pip install pennylane-lightning[gpu]  # has erro

import numpy as np
from pennylane.optimize import AdamOptimizer
from pennylane import numpy as qml_np
import pennylane as qml

import math
import matplotlib.pyplot as plt
import datetime


device_name = 'default.qubit'  #'default.qubit' # 
device_name2 = 'default.qubit' # has qml.state()


num_of_qubits = 1+4
eps_val_q = 1/math.sqrt(2**num_of_qubits)/100
eps_val = min(1e-10, eps_val_q)
tiny_change_threshold = 1e-4
cnt_threshold_no_change = 5

N = 2**(num_of_qubits-2)
normal_val = math.sqrt(1/N)
# initial_state_phi1 = [math.sqrt(1/N)]*(N-1) + [0]*N + [math.sqrt(1/N)] # 2**(num_of_qubits-1)
# initial_state_phi1 = [.5,.5,.5, 0,   0, 0, 0, .5,  ] # 2**(num_of_qubits-1)

# initial_state2 = [1/math.sqrt(N)]*(N-2) + [0, 1/math.sqrt(N)] + [0]*(N-2) + [1/math.sqrt(N), 0] # 2**(num_qubits-1)
# start_state = np.array([0.6, 0.3, 0.0, 0.1, 0.0, 0.0, 0.5, 0.5])
start_state = np.array([0.6, 0.3, 0.0, 0.1, 0.0, 0.0,0.0,0.2,0.0,0.4,0.3,0.0,0.0,0.5, 0.0, 0.0])
start_state = start_state/np.linalg.norm(start_state)
initial_state2 = start_state.tolist()

initial_state_phi1 = initial_state2
print(f'initial_state_phi1={initial_state_phi1}')
# print(f'initial_state_phi1={initial_state_phi1[-5:]}')
initial_state_0_phi1  = initial_state_phi1 + [0]*len(initial_state_phi1) # 2**num_of_qubits


# print(f'initial_state3={initial_state3}')
# initial_state  = initial_state3 + [0]*len(initial_state3) # 2**num_qubits





def layer_t3_no_HT(theta, qubit_posi):
    # type-2 layer
    # length of theta: (num_of_qubits-1)*2
    # length of qubit_posi: num_of_qubits-1
    # number of wires: num_of_qubits
    for i in range(num_of_qubits-1):
        qml.RY(theta[i], wires=(qubit_posi[i]))    
    for i in np.arange(0, num_of_qubits-2, 2):
#         qml.ctrl(qml.PauliZ(qubit_posi[i+1]), qubit_posi[i]) # CZ struct2
        qml.CNOT(wires=(qubit_posi[i],qubit_posi[i+1])) # CNOT struct3
    for i in range(num_of_qubits-1):
        qml.RY(theta[i+num_of_qubits-1], wires=(qubit_posi[i]))
    for i in np.arange(1, num_of_qubits-2, 2):
#         qml.ctrl(qml.PauliZ(qubit_posi[i+1]), qubit_posi[i]) # CZ struct2
        qml.CNOT(wires=(qubit_posi[i],qubit_posi[i+1])) # CNOT struct3
#     qml.ctrl(qml.PauliZ(qubit_posi[0]), qubit_posi[-1]) # CZ struct2
    qml.CNOT(wires=(qubit_posi[-1],qubit_posi[0])) # CNOT struct3
        
def layer_t3_with_HT(theta, num_of_qubits):
    # type-2 layer
    # length of theta: (num_of_qubits-1)*2
    # number of wires: num_of_qubits
    for i in range(num_of_qubits-1):
        qml.CRY(theta[i], wires=(0, i+1))    
    for i in np.arange(0, num_of_qubits-2, 2):
#         qml.ctrl(qml.PauliZ(i+2), (0, i+1)) # CZ struct2
        qml.Toffoli(wires=(0,i+1,i+2)) # CCNOT struct3
        
    for i in range(num_of_qubits-1):
        qml.CRY(theta[i+num_of_qubits-1], wires=(0, i+1))
    for i in np.arange(1, num_of_qubits-2, 2):
#         qml.ctrl(qml.PauliZ(i+2), (0, i+1)) # CZ struct2
        qml.Toffoli(wires=(0,i+1,i+2)) # CCNOT struct3
#     qml.ctrl(qml.PauliZ(1), (0, num_of_qubits-1)) # CZ struct2
    qml.Toffoli(wires=(0,num_of_qubits-1, 1)) # CCNOT struct3
    
test_flag = False  # True # False
if test_flag:
    for num_of_qubits in [10, 11]:   # 10 or 11 for the test below
        print('num_of_qubits=', num_of_qubits)
        dev_with_HT=qml.device(device_name, wires=num_of_qubits)
        @qml.qnode(dev_with_HT)
        def quantum_circuit_test3(theta):
            qubit_posi = list(range(math.floor(len(theta)/2)))
            layer_t3_no_HT(theta, qubit_posi)
            # length of theta: (num_of_qubits-1)*2
            # number of wires: num_of_qubits
            return qml.expval(qml.PauliZ(0))

        dev_with_HT=qml.device(device_name, wires=num_of_qubits)
        @qml.qnode(dev_with_HT)
        def quantum_circuit_test4(theta):
            qubit_posi = list(range(math.floor(len(theta)/2)))
            layer_t3_with_HT(theta, num_of_qubits)
            # length of theta: (num_of_qubits-1)*2
            # number of wires: num_of_qubits
            return qml.expval(qml.PauliZ(0))

        if num_of_qubits == 10:
            theta = [1,2,3,4,5,6,7,8,9,  1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1]
            test1_val = quantum_circuit_test3(theta)
            print(qml.draw(quantum_circuit_test3)(theta))
            print('test3_val=', test1_val)

            theta = [1,2,3,4,5,6,7,8,9,  1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1]
            test1_val = quantum_circuit_test4(theta)
            print(qml.draw(quantum_circuit_test4)(theta))
            print('test4_val=', test1_val)

        if num_of_qubits == 11:
            theta = [1,2,3,4,5,6,7,8,9,10,  1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1]
            test1_val = quantum_circuit_test3(theta)
            print(qml.draw(quantum_circuit_test3)(theta))
            print('test3_val=', test1_val)

            theta = [1,2,3,4,5,6,7,8,9,10,  1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1]
            test1_val = quantum_circuit_test4(theta)
            print(qml.draw(quantum_circuit_test4)(theta))
            print('test4_val=', test1_val)
            
            
# dev_with_HT=qml.device(device_name2, wires=num_of_qubits+1) #AerDevice(wires=num_of_qubits, shots=20000, backend='qasm_simulator')
dev_with_HT=qml.device(device_name, wires=num_of_qubits)
@qml.qnode(dev_with_HT)
def quantum_circuit_with_HT(theta):
    # initiate state vector |phi_1>
    qml.QubitStateVector(np.array(initial_state_0_phi1), wires=range(num_of_qubits))
#     qubit_position = list(range(1,num_of_qubits))
#     initiate_state_0_phi1(qml, qubit_position, work_wires=num_of_qubits)
    qml.Hadamard(0)
    for theta_i in theta:
        layer_t3_with_HT(theta_i, num_of_qubits)
    qml.Hadamard(0)    
    return qml.expval(qml.PauliZ(0)) 
    # return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))
print(qml.draw(quantum_circuit_with_HT)([[0.1]*2*(num_of_qubits-1)]))
# print(quantum_circuit_with_HT([[0.1]*(num_of_qubits-1)]))



# dev_with_HTZ=qml.device(device_name2, wires=num_of_qubits+1) #AerDevice(wires=num_of_qubits, shots=20000, backend='qasm_simulator')
dev_with_HTZ=qml.device(device_name, wires=num_of_qubits)
@qml.qnode(dev_with_HTZ)
def quantum_circuit_with_HTZ(theta):
    # initiate state vector |phi_1>
    qml.QubitStateVector(np.array(initial_state_0_phi1), wires=range(num_of_qubits))
#     qubit_position = list(range(1,num_of_qubits))
#     initiate_state_0_phi1(qml, qubit_position, work_wires=num_of_qubits)
    qml.Hadamard(0)
    for theta_i in theta:
        layer_t3_with_HT(theta_i, num_of_qubits)
    qml.CZ([0,1])
    qml.Hadamard(0)    
    return qml.expval(qml.PauliZ(0)) 
    # return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))
print('newly added')
print(qml.draw(quantum_circuit_with_HTZ)([[0.1]*2*(num_of_qubits-1)]))
# print(quantum_circuit_with_HTZ([[0.1]*(num_of_qubits-1)]))



# dev_no_HT_Z=qml.device(device_name2, wires=num_of_qubits+1) #AerDevice(wires=num_of_qubits-1, shots=20000, backend='qasm_simulator')
dev_no_HT_Z=qml.device(device_name, wires=num_of_qubits-1)        
@qml.qnode(dev_no_HT_Z)
def quantum_circuit_no_HT_return_Z(theta):
    # initiate state vector |phi_1>
    qml.QubitStateVector(np.array(initial_state_phi1), wires=range(num_of_qubits-1))
#     qubit_position = list(range(num_of_qubits-1))
#     initiate_state_0_phi1(qml, qubit_position, work_wires=num_of_qubits-1)
    for theta_i in theta:
        layer_t3_no_HT(theta_i, list(range(num_of_qubits-1)))
    
    return qml.expval(qml.PauliZ(0))  
    # return qml.sample(qml.PauliZ(0)) 
print('newly added 2')   
print(qml.draw(quantum_circuit_no_HT_return_Z)([[0.2]*2*(num_of_qubits-1)]))
# print(quantum_circuit_with_HT([[0.2]*(num_of_qubits-1)]))


# dev_no_HT_S=qml.device(device_name2, wires=num_of_qubits+1) #AerDevice(wires=num_of_qubits-1, backend='qasm_simulator')
dev_no_HT_S=qml.device(device_name2, wires=num_of_qubits-1)  
@qml.qnode(dev_no_HT_S)
def quantum_circuit_no_HT_return_state(theta):
    # initiate state vector |phi_1>
    qml.QubitStateVector(np.array(initial_state_phi1), wires=range(num_of_qubits-1))
#     qubit_position = list(range(num_of_qubits-1))
#     initiate_state_0_phi1(qml, qubit_position, work_wires=num_of_qubits-1)
    for theta_i in theta:
        layer_t3_no_HT(theta_i, list(range(num_of_qubits-1)))    
    return qml.state()

print('newly added 3')


val_global = []
coef2 = 1
def objective_fn(theta):
    global val_global
    val1_1 = quantum_circuit_with_HT(theta)
    val1_2 = quantum_circuit_with_HTZ(theta)
    val1_1 = val1_1/normal_val
    val1_2 = val1_2/normal_val
#     val2 = quantum_circuit_no_HT_return_Z(theta)
#     return coef2*val2-0.5*(val1_1 - val1_2)
    obj = -0.5*(val1_1 - val1_2)
    val_global.append([val1_1._value.tolist(), val1_2._value.tolist(), obj._value.tolist()])
    return obj




max_repeat = 1 #100
iter_max = 300  #300
num_of_layers = 3
prb_last_list = []
obj_list_rep = []
theta_list = []
iter_terminate_list = []
debug_flag, print_flag = 0, True
start_time = datetime.datetime.now()

for rep in range(1,max_repeat+1):
    if print_flag:
        print(f'\n\nrep={rep}')
    else:
        print(f'\n\nrep={rep}', end='  ')
    
    optimizer = AdamOptimizer(0.05, beta1=0.9, beta2=0.999)
    theta=qml_np.random.uniform(0, 2*math.pi, size=(num_of_layers, 2*(num_of_qubits-1)), requires_grad=True)
    obj_list = []
    tiny_change_cnt = 0
    break_flag = False
    iter_terminate=iter_max
    for iter in range(1, iter_max+1):    
        theta, obj = optimizer.step_and_cost(objective_fn, theta)
#         val1_1 = quantum_circuit_with_HT(theta)
#         val1_2 = quantum_circuit_with_HTZ(theta)
#         val2 = quantum_circuit_no_HT_return_Z(theta)
        val1_1 = val_global[-1][0]
        val1_2 = val_global[-1][1]
#         val2 = val_global[-1][2]
        if iter>=2:
            val1_1_old = val_global[-2][0]
            val1_2_old = val_global[-2][1]
            #val2_old = val_global[-2][2]
        else:
            val1_1_old = 999
            val1_2_old = 999
            #val2_old = 999
        val1 = val1_1 - val1_2
        val1_old = val1_1_old - val1_2_old
        if abs(val1) > eps_val: # eps_val=1e-10
            if abs( (val1-val1_old)/val1 ) < tiny_change_threshold: # 1e-3
                tiny_change_cnt += 1
            else:
                tiny_change_cnt = 0
        if tiny_change_cnt >= cnt_threshold_no_change: # no change for a consequtive of 5 iterations, then break
            break_flag = True
        if (iter==1 or iter%50==0 or iter==iter_max) and print_flag:
            print(f'iter={iter:3d} :: obj={obj:12.8f} :: val1_1={val1_1:12.8f} :: val1_2={val1_2:12.8f} :: -0.5*(val1_1 - val1_2)={-0.5*(val1_1 - val1_2):12.8f}')
            # print(f'iter={iter:3d} :: obj={obj:12.8f} :: val1_1={val1_1:12.8f} \
            #      :: val1_2={val1_2:12.8f} :: -0.5*(val1_1 - val1_2)={-0.5*(val1_1 - val1_2):12.8f}  \
            #      :: theta={theta}')

        obj_list.append(obj)
        if break_flag:
            iter_terminate=iter
            break
    theta_list.append(theta)
    # print(f'obj_list(last 3)={obj_list[-3:]}')
    obj_list_rep.append(obj_list)
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    duration_in_s = duration.total_seconds()
    print(f'time consumed: {duration_in_s}s, after for-iter')
    
    # print('val_global=')
    # print(val_global)
    val_global = [] # reset to empty
    
    ## display the amplified state
    state = quantum_circuit_no_HT_return_state(theta)
    prb = [i.item()**2 for i in state]

    # print(f'state={state}')
    if len(prb)>20:
        print(f'prb(last 2)={prb[-2:]}')
    else: print(f'prb={prb}')
    iter_terminate_list.append(iter_terminate)
    prb_last_list.append(prb[-2])
# print('theta_list=', theta_list)
# print('iter_terminate_list=', iter_terminate_list)
# print('prb_last_list=', prb_last_list)
end_time = datetime.datetime.now()
duration = end_time - start_time
duration_in_s = duration.total_seconds()
print(f'time consumed: {duration_in_s}s')



print('time now: ', end_time)
print('prb_last_list = ',prb_last_list)
# print('theta_list=', theta_list)
print('iter_terminate_list=', iter_terminate_list)