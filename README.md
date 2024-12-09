# msi2slstr-training-lightning-implementation
The **public** training environment for the `msi2slstr` model _(pytorch lightning version)_.

#### Notes:
1. _Expects a `data` folder in project directory to be pointing to a `msi2slstr-datagen` generated data directory._
2. _To deploy a model version to the `msi2slstr` production repo, export an onnx model 
to the `exports` folder using the `export_onnx.py` script and create a pull request._
