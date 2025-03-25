import torch
import copy

# Path to original SAM weight, for example: xxx/sam_vit_large/pytorch_model.bin
model_weight_path = 'xxx/sam_vit_large/pytorch_model.bin'

# Path to new SAM weight, for example: xxx/sam_vit_large/pytorch_model_new.bin
model_new = 'xxx/sam_vit_large/pytorch_model_new.bin'


model = torch.load(model_weight_path)
new_model = {}

for k, v in model.items():
    if k.startswith('mask_decoder.'):
        # deep copy v
        v_copy = copy.deepcopy(v)
        new_model[k] = v
        k_word = k.replace('mask_decoder.', 'mask_decoder_bo.')
        new_model[k_word] = v_copy
    else:
        new_model[k] = v

torch.save(new_model, model_new)

new_model = torch.load(model_new)
print('Old Weight: ')
print(len(model.keys()))
print(model.keys())

print('-------------------------------------------')
print('New Weight: ')
print(len(new_model.keys()))
print(new_model.keys())
print('-------------------------------------------')
print('Convert Done!')