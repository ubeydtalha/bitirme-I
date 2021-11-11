from decida.SimulatorNetlist import SimulatorNetlist

s = SimulatorNetlist("sar_seq_dig.net", simulator="ngspice")
print ("subcircuits :")
print( s.get("subckts"))
print ("instances :")
print( s.get("insts"))
print ("capacitors:")
print (s.get("caps"))
print ("resistors:")
print (s.get("ress"))