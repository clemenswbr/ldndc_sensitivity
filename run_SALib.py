from SALib.sample import sobol as sobolsample
from mpi4py import MPI
import configparser
import subprocess
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import sys

if __name__ == '__main__':
    
	print('START')
    
	##Configure site input
	config = configparser.ConfigParser()
	config_file = config.read(sys.argv[-1])
	print(f'Reading config file {sys.argv[-1]}')

	site = config.get('SITE', 'site_name')
	landuse = config.get('SITE', 'landuse')
	path = config.get('SITE', 'path_to_site')
	site = path.split('/')[-1]
	
	##MPI configuration
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	##Read parameter ranges (also necessary for running simulations)
	SALib_file_par_ranges = config.get('SALib_SETUP', 'SALib_path_par_ranges')
	dfp = pd.read_csv(SALib_file_par_ranges, sep='\t')

	##Get parameters
	SALib_file_par_values = config.get('SALib_SETUP', 'SALib_path_parameter_values')

	if SALib_file_par_values:
		print(f'Reading {SALib_file_par_values}')
		parameter_values = np.loadtxt(SALib_file_par_values, float)

	elif rank == 0:
		print('Start sampling...')

		n = 2**int(config.get('SALib_SETUP', 'n_reps'))

		problem = {
		'num_vars': len(dfp),
		'names': dfp['name'].values,
		'bounds': [[dfp[dfp['name'] == para]['min'].values[0], dfp[dfp['name'] == para]['max'].values[0]] for para in dfp['name']]
		}
		
		parameter_values = sobolsample.sample(problem, n)
		np.savetxt('SALib_parameter_values.txt', parameter_values)

		print(f'DONE sampling {len(parameter_values)} parameter sets for {len(dfp)} parameters')

		for i in range(1, size):
			
			comm.send(parameter_values, dest=i)

	else:
		parameter_values = comm.recv(source=0)

	##Create setup file
	output_setup = str(config.get('LDNDC_SETUP', 'ldndc_output_setup'))

	if not os.path.exists(f'{path}/{site}_sa_setup.xml'):
		#Setup file
		setupfilein = f'{path}/{site}_setup.xml'
		setupfileout = f'{path}/{site}_sa_setup.xml'

		setupfile = ET.parse(setupfilein)
		root = setupfile.getroot()
		modulelist = root.find('setup/mobile/modulelist')

		elements = []

		for module in modulelist:

			if module.attrib['id'].startswith('output'):
				elements.append(module)

		for e in elements:
			
			modulelist.remove(e)

		sc_output = ET.SubElement(modulelist, 'module')
		sc_output.set('id', output_setup)

		ET.indent(setupfile)
		setupfile.write(setupfileout)
			
	##Split parameters according to rank
	try:
		parameter_values = np.split(parameter_values, np.arange(len(parameter_values) // size, len(parameter_values) // size * size, len(parameter_values) // size))[rank]

	except ZeroDivisionError:
		print('Number of simulations < number of ranks; less than 1 simulation per rank \n Use less ranks or sample more parameters')
		
	##Create new siteparameter and project files, run simulation, read output, write output to concatenated file
	for run in range(len(parameter_values)):

		top = ET.Element('ldndcsiteparameters')
		pars = ET.SubElement(top, 'siteparameters')
		pars.set('id', str(0))

		if len(dfp) > 1:
			parameters = parameter_values[run]
		else:
			parameters = [parameter_values[run]]

		for name, value in zip(dfp['name'], parameters):
		
			par = ET.SubElement(pars, 'par')
			par.set('name', name)
			par.set('value', str(round(value, 8))) #Round sampled values from SALib

			tree = ET.ElementTree(top)
			ET.indent(tree)

			sparfileout = f'{path}/{site}__SALib_{rank}_{run}__siteparameters.xml'

			tree.write(sparfileout, encoding='utf-8', xml_declaration=True)
		
		##Create new project file
		treat_name = config.get('SITE', 'treat_name')

		pfilein = f'{path}/{site}{treat_name}.ldndc'
		pfileout = f'{path}/{site}_{treat_name}_SALib_{rank}_{run}.ldndc'

		pfile = ET.parse(pfilein)
		root = pfile.getroot()
		
		config_time = config.get('LDNDC_SETUP', 'time')

		if len(config_time) > 0:

			time = root.find('schedule')
			time.set('time', config_time)

		sps = root.find('input/sources/siteparameters')
		sps.set('source', f'_SALib_{rank}_{run}__siteparameters.xml')

		setup = root.find('input/sources/setup')
		setup.set('source', 'sa_setup.xml')

		sink = root.find('output/sinks')
		sink.set('sinkprefix', f'{landuse}/{site}/{site}_SALib_output/{site}_{treat_name}_SALib_{rank}_{run}_')

		pfile.write(pfileout)

		##Run simulation
		ldndc_executable = str(config.get('LDNDC_SETUP', 'ldndc_path_executable'))
		ldndc_conf_file = str(config.get('LDNDC_SETUP', 'ldndc_path_conf_file'))

		if eval(config.get('LDNDC_SETUP', 'dry_run')):
			cmd = f'echo {ldndc_executable} -c {ldndc_conf_file} {pfileout}'

		else:
			cmd = f'{ldndc_executable} -c {ldndc_conf_file} {pfileout}'
			print(cmd)
		
		p = subprocess.run(cmd.split())
		
		##Read inputs and write to concatenated file for each rank
		delete_outputs = eval(config.get('LDNDC_SETUP', 'delete_outputs'))
		output_file = str(config.get('LDNDC_SETUP', 'ldndc_output_file'))

		try:
			sc_out = pd.read_csv(f'{path}/{site}_SALib_output/{site}_{treat_name}_SALib_{rank}_{run}_{output_file}', sep = '\t')
			sc_out['run_index'] = f'{rank}_{run}'

			if (sc_out.to_numpy().flatten() == -99.99).any() or (len(sc_out) == 0):
				print(f'Corrupt soilchemistry file! -> {path}/{site}_SALib_output/{site}_{treat_name}_SALib_{rank}_{run}_{output_file}')

			else:
				#Write to output file
				if os.path.exists(f'{path}/{site}_SALib_output/{site}_SALib_{output_file}'):
					sc_out.to_csv(f'{path}/{site}_SALib_output/{site}_SALib_{output_file}', sep='\t', mode='a', header=False)
			
				else:
					sc_out.to_csv(f'{path}/{site}_SALib_output/{site}_SALib_{output_file}', sep='\t', header=True)

				if delete_outputs:
					sc_out_name = f'{path}/{site}_SALib_output/{site}_{treat_name}_SALib_{rank}_{run}_{output_file}'
					log_name = f'{path}/{site}_SALib_output/{site}_{treat_name}_SALib_{rank}_{run}_ldndc.log'
					subprocess.run(f'rm {sc_out_name} {log_name} {pfileout} {sparfileout}'.split())
			
		except FileNotFoundError:
			print(f'File not found! -> {path}/{site}_SALib_output/{site}_{treat_name}_SALib_{rank}_{run}_{output_file}')

print('DONE')