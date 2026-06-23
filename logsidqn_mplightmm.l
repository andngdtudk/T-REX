Loading net-file from '/home/s243292/pmt/T-REX/environments/kbh_j1_442m/kbh_j1_4.4.2.net.xml' ... done (11ms).
Loading additional-files from '/home/s243292/pmt/T-REX/environments/kbh_j1_442m/vtypes.xml' ... done (0ms).
Loading route-files incrementally from '/home/s243292/pmt/T-REX/environments/kbh_j1_442m/routes_car.rou.xml'
Loading route-files incrementally from '/home/s243292/pmt/T-REX/environments/kbh_j1_442m/routes_bike.rou.xml'
Loading route-files incrementally from '/home/s243292/pmt/T-REX/environments/kbh_j1_442m/routes_ped.rou.xml'
Loading done.
Simulation version 1.26.0 started via libsumo with time: 0.00.
Using libsumo as traci as requested by environment variable.
kbh_j1_442m /home/s243292/pmt/T-REX/environments/kbh_j1_442m/kbh_j1_4.4.2.sumocfg mplight_mm mplight_mm
lights 1 ('J01',)
GENERATING CONFIG
'J01': {
'lane_sets':{'S-W': ['01.01#S0_0', '01.01#S0_1'], 'S-S': ['01.01#S0_2', '01.01#S0_3', '01.01#O0_1'], 'S-E': ['01.01#O0_1', '01.01#O0_2'], 'W-N': ['01.01#O0_2', '01.01#O0_3', '01.01#O0_4'], 'W-W': ['01.01#O0_5', '01.01#O1.203_0'], 'W-S': ['01.01#O1.203_0', '01.01#N0_1'], 'N-E': ['01.01#N0_1', '01.01#N0_2', '01.01#N0_3'], 'N-N': ['01.01#N0_4', '01.01#E0_1'], 'N-W': ['01.01#E0_1', '01.01#E0_2'], 'E-S': ['01.01#E0_3', '01.01#E1_0'], 'E-E': ['01.01#E2_0'], 'E-N': ['01.02#S0_1', '01.02#S0_2']},
'downstream':{'N': None, 'E': None, 'S': None, 'W': None}},
[J01] derived 8 movements from phase green-signatures: {0: (1, 0, 0, 0), 1: (0, 1, 0, 0), 2: (1, 1, 0, 1), 3: (0, 0, 1, 0), 4: (1, 0, 1, 0), 5: (0, 0, 0, 1), 6: (1, 0, 0, 1), 7: (0, 1, 1, 0)}
[J01] movement -> inbound lanes: {0: ['01.01#S0_0', '01.01#S0_1', '01.01#N0_1', '01.01#N0_3'], 1: ['01.01#S0_2', '01.01#S0_3', '01.01#O1.203_0', '01.01#N0_4', '01.01#E2_0'], 2: ['01.01#O0_1'], 3: ['01.01#O0_1', '01.01#O0_2', '01.01#O0_3', '01.01#O0_4', '01.01#O1.203_0', '01.01#E0_1', '01.01#E0_2', '01.01#E0_3', '01.01#E2_0'], 4: ['01.01#O0_1', '01.01#N0_1', '01.01#E0_1'], 5: ['01.01#O0_5', '01.01#O1.203_0', '01.01#E1_0', '01.01#E2_0'], 6: ['01.01#N0_2', '01.02#S0_1'], 7: ['01.02#S0_2']}
[J01] movement -> outbound lanes: {0: ['01.01#s0_1', '01.01#e0_1', '01.01#s0_3', '01.01#n0_1', '01.01#n1_0'], 1: ['01.01#e0_2', '01.01#e0_3', '01.01#n1_0', '01.01#o0_2', '01.01#s0_2'], 2: ['01.01#n0_1'], 3: ['01.01#o0_0', '01.01#n0_1', '01.01#n1_0', '01.01#o0_1', '01.01#o0_2', '01.01#o1_0', '01.01#s0_3', '01.01#e0_1', '01.01#e0_2', '01.01#e0_3', '01.01#e1_0'], 4: ['01.01#s0_1', '01.01#o0_0', '01.01#n0_1'], 5: ['01.01#s0_3', '01.01#s0_2', '01.01#n0_1', '01.01#n1_0'], 6: ['01.01#e0_2', 'E3_0'], 7: ['01.02#s0_2']}
[J01] >>> VERIFY THIS against the real intersection layout before trusting training results — there is no way to confirm from code alone that this ordering matches what phase_pairs' indices were intended to mean.
Generated signal config file: /home/s243292/pmt/T-REX/TREX_comp/config/generated/kbh_j1_442m_signal_config.py
