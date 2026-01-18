GEN_SYS_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PRE_SYS_PROMPT = "You are a helpful assistant that performs time series prediction. The user will provide a sequence and you will predict the sequence."
ANA_SYS_PROMPT = "You are a helpful assistant that performs time series analysis. The user will provide a sequence and you will respond to the questions based on this sequence."
INCONTEXT_SYS_PROMPT = ""

PRE_INST_PROMPT = "Please predict the following sequence carefully."
PRE_INST_PROMPT_TEXT = "Please predict the following sequence carefully. Context knowledge you may consider: {}"
ANA_INST_PROMPT_TEXT = "Please answer the following question carefully after analyzing the sequence: {}"
INCONTEXT_PROMPT = ""


TEMPLATE = """{}

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_incontext(examples, test_instruction, test_input, response):
    """
    examples: list of tuples (instruction, input, response)
    test_instruction: str
    test_input: str
    """
    parts = []
    for e in examples:
        # parse the input
        inp = e.split("Input:")[1].split("Response:")[0].strip()
        resp = e.split("Response:")[1].strip()
        parts.append(f"""### Input:
{inp}

### Response:
{resp}
""")
    # Now append the test query (no response filled in yet)
    if response is None:
        parts.append(f"""### Input:
{test_input}

### Response:""")
    else:
        parts.append(f"""### Input:
{test_input}

### Response:{response}""")
    
    return "\n".join(parts)



def getPrompt(flag, instruction=None, input=None, response=None, context=None):
    if flag == "general":  # instruction, [input], [response]
        system = GEN_SYS_PROMPT
        if instruction is None:
            raise ValueError("Instruction must be provided for general tasks.")
        else:
            instruction = instruction
        input = "" if input is None else input
        response = "" if response is None else response

    elif flag == "prediction":  # [context], input, [response]
        system = PRE_SYS_PROMPT
        instruction = PRE_INST_PROMPT if context is None else PRE_INST_PROMPT_TEXT.format(context)
        if input is None:
            raise ValueError("Input must be provided for prediction tasks.")
        else:
            input = input
        response = "" if response is None else response

    elif flag == "incontext":  # instruction, [context], input, [response]
        system = PRE_SYS_PROMPT + "\n\n### Instruction:\n" + PRE_INST_PROMPT
        if context is None:
            instruction = PRE_INST_PROMPT 
        else:
            prompt = format_incontext(context, PRE_INST_PROMPT, input, response)
            return system + "\n\n" + prompt

    elif flag == "analysis":  # instruction, input, [response]
        system = ANA_SYS_PROMPT
        if instruction is None:
            raise ValueError("Instruction must be provided for analysis tasks.")
        else:
            instruction = ANA_INST_PROMPT_TEXT.format(instruction)
        if input is None:
            raise ValueError("Input must be provided for analysis tasks.")
        else:
            input = input
        response = "" if response is None else response

    else:
        raise ValueError("Flag must be one of 'general', 'prediction', or 'analysis'.")

    prompt = TEMPLATE.format(system, instruction, input, response)

    return prompt

def getIncontextPrompt(context):
    return INCONTEXT_PROMPT.format(context)

def addInstruct(prompt):
    system = GEN_SYS_PROMPT
    instruction = PRE_INST_PROMPT
    prompt = TEMPLATE.format(system, instruction, prompt, "")
    return prompt