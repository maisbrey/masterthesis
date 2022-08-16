## codes

The **codes** folder contains the used quantum codes in the ALIST file format (see http://www.inference.org.uk/mackay/codes/alist.html for further information).

## modules

The **modules** folder contains several Python files to generate quantum codes, decode and evaluate them:
- **code_generation.py**: Contains code to generate quantum codes from BCH codes and from classical regular LDPC codes free of cycles of length 4. Examples of how to 
use it to generate the [[129, 28]] HP code and the [[900, 36]] HP code are provided in the notebook **generate_hp_codes.ipynb** in the **demos** folder.
- **pcm_management.py**: Contains the QuantumCode class, which enables to convert between different representations of a quantum code. An example of how to use it to 
write an ALIST file from a numpy array is given in the notebook **generate_hp_codes.ipynb** in the **demos** folder. An example of how to use it to read from an 
ALIST file is given in the notebook basics_of_bp_decoding.ipynb in the **demos** folder.
- **utils.py**: Contains several functions to generate certain error patterns as well as functions to optimize the update schedule of a serial BP4 decoder.
- **bp_decoder_bin_reg.py**: Contains the classes DecoderBinaryParallel and DecoderBinarySerial, which efficiently perform message passing decoding of regular classical
LDPC codes over GF2, using a parallel and a serial update schedule, respectively.
- **bp_decoder_hp_reg.py**: Contains the classes DecoderParallel and DecoderSerial, which efficiently perform message passing decoding of regular quantum
LDPC codes over GF4, using a parallel and a serial update schedule, respectively.
- **bp_decoder_hp_ireg.py**: Contains the classes DecoderParallel and DecoderSerial, which efficiently perform message passing decoding of irregular quantum
LDPC codes over GF4, using a parallel and a serial update schedule, respectively.

Note that more parallelization during decoding is possible if the underlying HP code is regular (hence the two different Python files for regular and irregular HP codes).
This is demonstrated and elaborated on in the notebook **basics_of_bp_decoding.ipynb** in the **demos** folder.
