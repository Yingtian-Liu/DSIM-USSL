# contains utility functions
import zipfile

def extract(source_path, destination_path):
    """function extracts all files from a zip file at a given source path
    to the provided destination path"""
    
    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)
    
    
    

class standardize:
    def __init__(self, mean_val=None,std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):        
        return (x - self.mean_val)/ self.std_val

    def unnormalize(self, x):
        return x*self.std_val + self.mean_val





    
# def standardize(seismic, model, no_wells):
#     """function standardizes data using statistics extracted from training 
#     wells
    
#     Parameters
#     ----------
#     seismic : array_like, shape(num_traces, depth samples)
#         2-D array containing seismic section
        
#     model : array_like, shape(num_wells, depth samples)
#         2-D array containing model section
        
#     no_wells : int,
#         no of wells (and corresponding seismic traces) to extract.
#     """
        
    
#     seismic_normalized = (seismic - seismic.mean())/ seismic.std()
#     train_indices = (np.linspace(0, len(model)-1, no_wells, dtype=np.int_))
    
#     # model_normalized = (model - model.mean()) / model.std()
#     model_normalized = np.zeros((model.shape[0], model.shape[1] ,model.shape[2]))
    
#     for i in range(model.shape[1]):
#         model_normalized[:,i,:] = (model[:,i,:] - model[train_indices,i,:].mean()) / model[train_indices,i,:].std()
    
#     return seismic_normalized, model_normalized

