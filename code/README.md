This folder contains the code used in the thesis. 

## codes

The **codes** folder contains the used quantum codes in the ALIST file format (see http://www.inference.org.uk/mackay/codes/alist.html for further information).

## modules

The **modules** folder contains several Python files to generate quantum codes, decode and evaluate them:
- **code_generation.py**: Contains code to generate quantum codes from BCH codes and to generate classical regular LDPC codes free of cycles of length 4. Examples of how to 
use it to generate the [[129, 28]] HP code and the [[900, 36]] HP code are provided in the notebook **generate_hp_codes.ipynb** in the **demos** folder.
- **pcm_management.py**: Contains the QuantumCode class, which enables to convert between different representations of a quantum code. An example of how to use it to 
write an ALIST file is given in the notebook **generate_hp_codes.ipynb** in the **demos** folder. An example of how to use it to read from an ALIST file is given in the
notebook basics_of_bp_decoding.ipynb in the **demos** folder.
- **utils.py**:

The decoder and it's modified versions are implemented as classes
