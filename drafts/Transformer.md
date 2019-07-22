# Transformer? Attention?
## Theory and Model
### Encoder-Decoder architecture
### Attention
- can be used in different tasks (text, visual, voice ...)
- one stone(attention), two birds(parallelize(within attention layer) and long-range dependencies)
- 3 types of attention
- attention operation is essentially feature reconstruction
- **multi-head attention** VS convolution on multiple channels
	- Convolution: Different linear transformations by relative position
	- MHA: a weighted average 
### Why multiple layer of attention layers?
### Vector similarity
### Positional encoding
- why not positional index? extrapolate training samples
### point-wise FFN
### Mask
## Training tricks
### layer normalization
### residual connection
- Help gradient BP
- Residuals carry positional information to higher layers, among other information.
### warn-up learning rate
### regularization
- dropout
- layer normalization

## Resources
[Attention is all you need review]([https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html](https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html))
[The transformer - Attention is all you need]([https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XTEl6ugzZPY](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XTEl6ugzZPY))
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjU4OTk5NzQ0LC0xODY5MTc4MjYsMTM2OT
YzOTg0NCwtMTExNDg0MTI5MiwyMTI1NjQzNjUwLC0xNDYzMTUz
NDM3LC0yMDA3MzUzNzQ1LC0yMjc1NDExMjksLTEzMTU5MTUwNS
wxMjE5MDIzMDIxXX0=
-->