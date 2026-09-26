import numpy as np

nsims = 448
ndatas = 100

# dictionaries to store the averaged cmplx_bispec and individual estimates for calculation of the std dev 
cmplx_bispectra = {}
individual_evaluations = {}

#Load in 
for i in range(ndatas+1):
    for j in range(nsims):
        if i != j:
            try:
                L, temp = np.loadtxt(f"../data{i}/multipledata_cmplx_data{i}_simstart{j}.txt") #load next evaluation of cmplx bispec est into temp variable
                # Check if this is first file loaded into cmplx_bispectra
                if i not in cmplx_bispectra:
                    cmplx_bispectra[i] = temp
                else:
                    cmplx_bispectra[i] += temp
        
                # Check if this is first file loaded into indiv evaluations                       
                if i not in individual_evaluations:
                    individual_evaluations[i] = [temp] #creates a list of arrays
                else:
                    individual_evaluations[i].append(temp) #this saves each array into a list. Every element of the list individual_evaluation[1] stores an array corresponding to the jth simulation and 1st data
            except OSError:
                pass
        else:
            pass   

# Average
for i in cmplx_bispectra:
    cmplx_bispectra[i] /= nsims
    

#Make dictionary for the errors

errors = {}

for i in individual_evaluations:
    matrix_of_all_sims = np.array(individual_evaluations[i]) #This converts the list of arrays into a matrix where each row is one of the arrays in the original list
    errors[i] = np.std(matrix_of_all_sims, axis=0) #axis 0 is columns so this calculates the std deviation along the rows (down each column) ie for each column what is the std deviation - this will be std dev across sims for a specific multipole.


#Save results

for i in cmplx_bispectra.keys():
    print(i)
    # Ensure that both cmplx_bispectra and errors have data for this key
    if i in errors:
        data_to_save = np.column_stack((L, cmplx_bispectra[i], errors[i]))
        np.savetxt(f"cmplx_bispec_data{i}.txt", data_to_save)
    else:
        print(f"No error data for index {i}, skipping saving.")
