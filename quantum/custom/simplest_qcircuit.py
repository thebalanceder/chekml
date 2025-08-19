from qiskit import QuantumCircuit, transpile,Aer

qc=QuantumCircuit(2,2)

qc.h(0)
qc.cx(0,1)

qc.measure(0,0)
qc.measure(1,1)

sim=Aer.get_backend("aer_simulator")
transpiled=transpile(qc,sim)
result=sim.run(transpiled).result()
print(result.get_counts())
