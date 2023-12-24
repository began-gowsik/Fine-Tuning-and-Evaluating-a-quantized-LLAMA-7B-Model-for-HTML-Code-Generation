
# Fine-Tuning and Evaluating a LLM Model for HTML Code Generation
## ML Intern Assignment - figr.design

Objective:

Fine-tune the Falcon 7B/LLAMA 7B model or another open-source language model to generate HTML code from natural language prompts. Showcase your skills in model fine-tuning, addressing challenges, and basic API development.

Task Overview:
1. Model Selection:


     Making LLMs lighter with AutoGPTQ and transformers

    
    
    Reference: https://huggingface.co/blog/gptq-integration
    
    https://huggingface.co/blog/overview-quantization-transformers#:~:text=From%20the%20benchmark%2C%20we%20can,more%20details%20on%20these%20benchmarks.

    GPTQ is a neural network compression technique that enables    
    the efficient deployment of Generative Pretrained Transformers 
    (GPT).

    1.1) How Does GPTQ work?

    GPTQ employs a Layerwise Quantization algorithm that quantizes the weights of the LLM individually. It converts floating-point parameters into quantized integers to minimize error at the output.

    1.2) Comparison with bitsandbytes:

    GPTQ:
    Utilizes Integer quantization and an optimization procedure with input mini-batch for quantization.
    After quantization, weights can be stored and reused.
    Bitsandbytes:
    Supports various formats, including integer quantization.
    Lacks an optimization procedure with an input mini-batch, making it less precise than GPTQ.


    1.3) Quantize transformers model using auto-gptq,  transformers and optimum:
Explore two scenarios for using this integration:

1- Quantize a language model from scratch:

    code available: GPTQ_model_quantizing.ipynb

2- Load a Pre-Quantized Model from Hub:


    Finally the Quantized model used here: TheBloke/Llama-2-7b-Chat-GPTQ

2.  Data Set:

To fine-tune the model, we utilized the below dataset from Hugging Face Datasets
        
        ''ttbui/alpaca_webgen_html''

The dataset comprises pairs of natural language prompts and corresponding HTML code, making it suitable for training our HTML code generation model. It contains 528 rows

3.  Fine-Tuning:

Parameter Efficient Fine-Tuning (PEFT):

PEFT methods have emerged as an efficient approach to fine-tune pretrained LLMs while significantly reducing the number of trainable parameters. These techniques balance computational efficiency and task performance, making it feasible to fine-tune even the largest LLMs without compromising on quality.

Low-Rank Adaptation (LoRA):

LoRA is an innovative technique designed to efficiently fine-tune pre-trained language models by injecting trainable low-rank matrices into each layer of the Transformer architecture. LoRA aims to reduce the number of trainable parameters and the computational burden while maintaining or improving the model’s performance on downstream tasks.

Configuration of LORA Hyperparameters:

    r (int): Controls the rank of low-rank approximation in LORA, affecting long-range dependency modeling and computational cost

    alpha (float): Scales attention scores in LORA, adjusting relative importance of attention from different positions

    dropout (float): Applies dropout regularization to LORA layers to prevent overfitting

    target_modules (list of str): Specifies attention layers where LORA should be applied

    bias (str): Sets bias handling for LORA layers

    task_type (str): Specifies the type of task the model is being used for.

Configuration of Training:

    model: The pre-trained language model to be trained

    train_dataset: The training dataset to use for learning

    per_device_train_batch_size: Number of samples processed per device per training step

    gradient_accumulation_steps: Number of steps to accumulate gradients before updating model weights

    warmup_steps: Number of steps for learning rate warmup

    max_steps: Maximum number of training steps

    learning_rate: Base learning rate for model optimization

    fp16: Enables mixed-precision training for faster computation and reduced memory usage

    logging_steps: Frequency of logging training progress

    output_dir: Directory to save model checkpoints and training outputs

    optim: Optimizer to use (in this case, adamw_hf, a variant of AdamW)

    data_collator: Function for preparing batches of data for training (here, DataCollatorForLanguageModeling for language modeling tasks)

4. Evaluation:

During evaluation, the model generates predictions for instructions, and the script records the predicted outputs alongside the ground truth HTML code. The number of exact matches between predictions and ground truth is tallied and printed at the end of the evaluation loop. This serves as a key metric to assess the model's ability to accurately generate HTML code from given prompts, providing insights into the model's overall performance.

5. API development:






    







# Fine-Tuning and Evaluating a LLM Model for HTML Code Generation
## ML Intern Assignment - figr.design

Objective:

Fine-tune the Falcon 7B/LLAMA 7B model or another open-source language model to generate HTML code from natural language prompts. Showcase your skills in model fine-tuning, addressing challenges, and basic API development.

Task Overview:
1. Model Selection:


     Making LLMs lighter with AutoGPTQ and transformers

    
    
    Reference: https://huggingface.co/blog/gptq-integration
    
    https://huggingface.co/blog/overview-quantization-transformers#:~:text=From%20the%20benchmark%2C%20we%20can,more%20details%20on%20these%20benchmarks.

    GPTQ is a neural network compression technique that enables    
    the efficient deployment of Generative Pretrained Transformers 
    (GPT).

    1.1) How Does GPTQ work?

    GPTQ employs a Layerwise Quantization algorithm that quantizes the weights of the LLM individually. It converts floating-point parameters into quantized integers to minimize error at the output.

    1.2) Comparison with bitsandbytes:

    GPTQ:
    Utilizes Integer quantization and an optimization procedure with input mini-batch for quantization.
    After quantization, weights can be stored and reused.
    Bitsandbytes:
    Supports various formats, including integer quantization.
    Lacks an optimization procedure with an input mini-batch, making it less precise than GPTQ.


    1.3) Quantize transformers model using auto-gptq,  transformers and optimum:
Explore two scenarios for using this integration:

1- Quantize a language model from scratch:

    code available: GPTQ_model_quantizing.ipynb

2- Load a Pre-Quantized Model from Hub:


    Finally the Quantized model used here: TheBloke/Llama-2-7b-Chat-GPTQ

2.  Data Set:

To fine-tune the model, we utilized the below dataset from Hugging Face Datasets
        
        ''ttbui/alpaca_webgen_html''

The dataset comprises pairs of natural language prompts and corresponding HTML code, making it suitable for training our HTML code generation model. It contains 528 rows

3.  Fine-Tuning:

Parameter Efficient Fine-Tuning (PEFT):

PEFT methods have emerged as an efficient approach to fine-tune pretrained LLMs while significantly reducing the number of trainable parameters. These techniques balance computational efficiency and task performance, making it feasible to fine-tune even the largest LLMs without compromising on quality.

Low-Rank Adaptation (LoRA):

LoRA is an innovative technique designed to efficiently fine-tune pre-trained language models by injecting trainable low-rank matrices into each layer of the Transformer architecture. LoRA aims to reduce the number of trainable parameters and the computational burden while maintaining or improving the model’s performance on downstream tasks.

Configuration of LORA Hyperparameters:

    r (int): Controls the rank of low-rank approximation in LORA, affecting long-range dependency modeling and computational cost

    alpha (float): Scales attention scores in LORA, adjusting relative importance of attention from different positions

    dropout (float): Applies dropout regularization to LORA layers to prevent overfitting

    target_modules (list of str): Specifies attention layers where LORA should be applied

    bias (str): Sets bias handling for LORA layers

    task_type (str): Specifies the type of task the model is being used for.

Configuration of Training:

    model: The pre-trained language model to be trained

    train_dataset: The training dataset to use for learning

    per_device_train_batch_size: Number of samples processed per device per training step

    gradient_accumulation_steps: Number of steps to accumulate gradients before updating model weights

    warmup_steps: Number of steps for learning rate warmup

    max_steps: Maximum number of training steps

    learning_rate: Base learning rate for model optimization

    fp16: Enables mixed-precision training for faster computation and reduced memory usage

    logging_steps: Frequency of logging training progress

    output_dir: Directory to save model checkpoints and training outputs

    optim: Optimizer to use (in this case, adamw_hf, a variant of AdamW)

    data_collator: Function for preparing batches of data for training (here, DataCollatorForLanguageModeling for language modeling tasks)

4. Evaluation:

During evaluation, the model generates predictions for instructions, and the script records the predicted outputs alongside the ground truth HTML code. The number of exact matches between predictions and ground truth is tallied and printed at the end of the evaluation loop. This serves as a key metric to assess the model's ability to accurately generate HTML code from given prompts, providing insights into the model's overall performance.

(More parts of evaluation to be added)

5. API development:

(Yet to be done)







    






