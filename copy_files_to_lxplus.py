import os

versions = [f'v{i}' for i in range(42,61)]
commands = []
for version in versions:
  commands.append(f'scp *{version}* lxplus:/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_inputs/{version}')
command = ' && '.join(commands)
command += ' &'
print(command)
os.system(command)
