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


device_name = 'default.qubit'  # 'default.qubit' #
device_name2 = 'default.qubit'  # has qml.state()
print('sss')


class VQS:
    def __init__(self, num_of_qubits, oracle=None):
        self.oracle = oracle

        self.device_name = 'default.qubit'  # 'default.qubit'
        self.device_name2 = 'default.qubit'  # has qml.state()
        self.val_global = []
        n = 2**(num_of_qubits-2)
        self.normal_val = math.sqrt(1/n)

        self.num_of_qubits = 1+num_of_qubits
        self.eps_val_q = 1/math.sqrt(2**num_of_qubits)/100
        self.eps_val = min(1e-10, self.eps_val_q)
        self.tiny_change_threshold = 1e-4
        self.cnt_threshold_no_change = 5
        # self.dev_with_HT=qml.device(device_name2, wires=num_of_qubits+1) #AerDevice(wires=num_of_qubits, shots=20000, backend='qasm_simulator')
        self.dev_with_HT = qml.device(device_name, wires=num_of_qubits)
        # self.dev_with_HTZ=qml.device(device_name2, wires=num_of_qubits+1) #AerDevice(wires=num_of_qubits, shots=20000, backend='qasm_simulator')
        self.dev_with_HTZ = qml.device(device_name, wires=num_of_qubits)
        # dev_no_HT_Z=qml.device(device_name2, wires=num_of_qubits+1) #AerDevice(wires=num_of_qubits-1, shots=20000, backend='qasm_simulator')
        self.dev_no_HT_Z = qml.device(device_name, wires=num_of_qubits-1)
        # dev_no_HT_S=qml.device(device_name2, wires=num_of_qubits+1) #AerDevice(wires=num_of_qubits-1, backend='qasm_simulator')
        self.dev_no_HT_S = qml.device(device_name2, wires=num_of_qubits-1)

    def layer_t3_no_HT(self, theta, qubit_posi):
        # type-2 layer
        # length of theta: (num_of_qubits-1)*2
        # length of qubit_posi: num_of_qubits-1
        # number of wires: num_of_qubits
        for i in range(self.num_of_qubits-1):
            qml.RY(theta[i], wires=(qubit_posi[i]))
        for i in np.arange(0, self.num_of_qubits-2, 2):
            #         qml.ctrl(qml.PauliZ(qubit_posi[i+1]), qubit_posi[i]) # CZ struct2
            qml.CNOT(wires=(qubit_posi[i], qubit_posi[i+1]))  # CNOT struct3
        for i in range(self.num_of_qubits-1):
            qml.RY(theta[i+self.num_of_qubits-1], wires=(qubit_posi[i]))
        for i in np.arange(1, self.num_of_qubits-2, 2):
            #         qml.ctrl(qml.PauliZ(qubit_posi[i+1]), qubit_posi[i]) # CZ struct2
            qml.CNOT(wires=(qubit_posi[i], qubit_posi[i+1]))  # CNOT struct3
    #     qml.ctrl(qml.PauliZ(qubit_posi[0]), qubit_posi[-1]) # CZ struct2
        qml.CNOT(wires=(qubit_posi[-1], qubit_posi[0]))  # CNOT struct3

    def layer_t3_with_HT(self, theta, num_of_qubits):
        # type-2 layer
        # length of theta: (num_of_qubits-1)*2
        # number of wires: num_of_qubits
        for i in range(num_of_qubits-1):
            qml.CRY(theta[i], wires=(0, i+1))
        for i in np.arange(0, num_of_qubits-2, 2):
            #         qml.ctrl(qml.PauliZ(i+2), (0, i+1)) # CZ struct2
            qml.Toffoli(wires=(0, i+1, i+2))  # CCNOT struct3

        for i in range(num_of_qubits-1):
            qml.CRY(theta[i+num_of_qubits-1], wires=(0, i+1))
        for i in np.arange(1, num_of_qubits-2, 2):
            #         qml.ctrl(qml.PauliZ(i+2), (0, i+1)) # CZ struct2
            qml.Toffoli(wires=(0, i+1, i+2))  # CCNOT struct3
    #     qml.ctrl(qml.PauliZ(1), (0, num_of_qubits-1)) # CZ struct2
        qml.Toffoli(wires=(0, num_of_qubits-1, 1))  # CCNOT struct3

    def quantum_circuit_with_HT(self, theta):
        @qml.qnode(self.dev_with_HT)
        def _quantum_circuit_with_HT(theta):
            # initiate state vector |phi_1>
            # qml.QubitStateVector(np.array(initial_state_0_phi1),wires=range(self.num_of_qubits))  # Need
            qml.Hadamard(0)
            for theta_i in theta:
                print(theta_i)
                self.layer_t3_with_HT(theta_i, self.num_of_qubits)
            qml.Hadamard(0)
            return qml.expval(qml.PauliZ(0))
        # print(qml.draw(_quantum_circuit_with_HT)
        #       ([[0.1]*2*(self.num_of_qubits-1)]))
        return _quantum_circuit_with_HT(theta)

    def quantum_circuit_with_HTZ(self, theta):
        @qml.qnode(self.dev_with_HTZ)
        def _quantum_circuit_with_HTZ(theta):
            # initiate state vector |phi_1>
            # qml.QubitStateVector(np.array(initial_state_0_phi1), #Need
            #                     wires=range(num_of_qubits))
            qml.Hadamard(0)
            for theta_i in theta:
                self.layer_t3_with_HT(theta_i, self.num_of_qubits)
            qml.CZ([0, 1])
            qml.Hadamard(0)
            return qml.expval(qml.PauliZ(0))
        # print('newly added')
        # print(qml.draw(_quantum_circuit_with_HTZ)
        #       ([[0.1]*2*(self.num_of_qubits-1)]))
        return _quantum_circuit_with_HTZ(theta)

    def quantum_circuit_no_HT_return_Z(self, theta):
        @qml.qnode(self.dev_no_HT_Z)
        def _quantum_circuit_no_HT_return_Z(theta):
            # initiate state vector |phi_1>
            # qml.QubitStateVector(np.array(self.initial_state_phi1),  #Need
            #                     wires=range(self.num_of_qubits-1))
            for theta_i in theta:
                self.layer_t3_no_HT(theta_i, list(range(self.num_of_qubits-1)))
            return qml.expval(qml.PauliZ(0))
        # print('newly added 2')
        # print(qml.draw(self.quantum_circuit_no_HT_return_Z)
        #       ([[0.2]*2*(self.num_of_qubits-1)]))
        return _quantum_circuit_no_HT_return_Z(theta)

    def quantum_circuit_no_HT_return_state(self, theta):
        @qml.qnode(self.dev_no_HT_S)
        def _quantum_circuit_no_HT_return_state(theta):
            # initiate state vector |phi_1>
            # qml.QubitStateVector(np.array(initial_state_phi1),  #theta
            #                     wires=range(num_of_qubits-1))
            for theta_i in theta:
                self.layer_t3_no_HT(theta_i, list(range(self.num_of_qubits-1)))
            return qml.state()
        print('newly added 3')
        return _quantum_circuit_no_HT_return_state(theta)

    def objective_fn(self, theta):
        val1_1 = self.quantum_circuit_with_HT(theta)
        val1_2 = self.quantum_circuit_with_HTZ(theta)
        val1_1 = val1_1/self.normal_val
        val1_2 = val1_2/self.normal_val
    #     val2 = quantum_circuit_no_HT_return_Z(theta)
    #     return coef2*val2-0.5*(val1_1 - val1_2)
        obj = -0.5*(val1_1 - val1_2)
        self.val_global.append(
            [val1_1._value.tolist(), val1_2._value.tolist(), obj._value.tolist()])
        return obj

    def run(self, coef2=1, max_repeat=1, iter_max=300, num_of_layers=3, print_flag=True):
        coef2 = coef2
        max_repeat = max_repeat
        iter_max = iter_max
        num_of_layers = num_of_layers
        prb_last_list = []
        obj_list_rep = []
        theta_list = []
        iter_terminate_list = []
        print_flag = print_flag

        last_prbs = []  # added
        print('start')
        for rep in range(1, max_repeat+1):
            if print_flag:
                print(f'\n\nrep={rep}')
            else:
                print(f'\n\nrep={rep}', end='  ')

            optimizer = AdamOptimizer(0.05, beta1=0.9, beta2=0.999)
            theta = qml_np.random.uniform(
                0, 2*math.pi, size=(num_of_layers, 2*(self.num_of_qubits-1)), requires_grad=True)
            obj_list = []
            tiny_change_cnt = 0
            break_flag = False
            iter_terminate = iter_max
            for iter in range(1, iter_max+1):
                theta, obj = optimizer.step_and_cost(self.objective_fn, theta)
                val1_1 = val_global[-1][0]
                val1_2 = val_global[-1][1]
                if iter >= 2:
                    val1_1_old = val_global[-2][0]
                    val1_2_old = val_global[-2][1]
                else:
                    val1_1_old = 999
                    val1_2_old = 999
                    #val2_old = 999
                val1 = val1_1 - val1_2
                val1_old = val1_1_old - val1_2_old
                if abs(val1) > self.eps_val:  # eps_val=1e-10
                    if abs((val1-val1_old)/val1) < self.tiny_change_threshold:  # 1e-3
                        tiny_change_cnt += 1
                    else:
                        tiny_change_cnt = 0
                # no change for a consequtive of 5 iterations, then break
                if tiny_change_cnt >= self.cnt_threshold_no_change:
                    break_flag = True
                if (iter == 1 or iter % 50 == 0 or iter == iter_max) and print_flag:
                    print(
                        f'iter={iter:3d} :: obj={obj:12.8f} :: val1_1={val1_1:12.8f} :: val1_2={val1_2:12.8f} :: -0.5*(val1_1 - val1_2)={-0.5*(val1_1 - val1_2):12.8f}')

                obj_list.append(obj)
                if break_flag:
                    iter_terminate = iter
                    break
            theta_list.append(theta)
            obj_list_rep.append(obj_list)

            val_global = []  # reset to empty

            # display the amplified state
            state = self.quantum_circuit_no_HT_return_state(theta)
            prb = [i.item()**2 for i in state]
            iter_terminate_list.append(iter_terminate)
            prb_last_list.append(prb[-2])
            last_prbs = prb

        results = []
        for i in range(len(last_prbs)):
            if(np.linalg.norm(last_prbs[i]) > 0.01):
                results.append((i, last_prbs[i]))
        return results


vqs = VQS(num_of_qubits=4)

print(vqs.run())
