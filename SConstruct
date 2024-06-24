import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
import steamroller

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file (see it for an
# example, changing the number of folds).
vars = Variables("/home/sbacker2/projects/post_ocr_correction/custom.py")
vars.AddVariables(
   # ("OUTPUT_WIDTH", "", 5000),
     ("USE_GRID", "", False),
     ("IMAGE_LOCATION", "", "/home/sbacker2/projects/post_ocr_correction/data/images"),
     ("GPT_VERSION", "", ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]),
    ("API_KEY", "", ""),
   # ("PARAMETER_VALUES", "", [0.1, 0.5, 0.9]),
    ("DATASETS", "", [["/home/sbacker2/projects/post_ocr_correction/data/images", "/home/sbacker2/projects/post_ocr_correction/data/texts"]]),
    ("PROMPTS", "", [["prompt_0", "This is a historical text from a digitized archive. It has been created using optical character recognition, introducing numerous errors to a text that initially had none. without adding any new material, please correct the text by fixing the errors created by OCR"]]))

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    
    # Defining a bunch of builders (none of these do anything except "touch" their targets,
    # as you can see in the dummy.py script).  Consider in particular the "TrainModel" builder,
    # which interpolates two variables beyond the standard SOURCES/TARGETS: PARAMETER_VALUE
    # and MODEL_TYPE.  When we invoke the TrainModel builder (see below), we'll need to pass
    # in values for these (note that e.g. the existence of a MODEL_TYPES variable above doesn't
    # automatically populate MODEL_TYPE, we'll do this with for-loops).
    BUILDERS={
        "Perform_OCR" : Builder(
            action="python scripts/perform_ocr_pytesseract.py --input_file ${INPUT} --output_file ${TARGETS[0]}"
        ),
        "ComparePerformanceInitial" : Builder(
            action="python scripts/compare_performance.py --test_directory ${SOURCES[0]} --control_directory ${CONTROL} --output_file ${TARGETS[0]}"
        ),
        "GPTCorrect" : Builder(
            action="python scripts/gpt_correction.py --prompt ${PROMPT} --gpt_version ${MODEL_TYPE} --input ${SOURCES[0]} --output ${TARGETS[0]} --api_key ${API_KEY}"            
        ),
	"GPT4OCorrect" : Builder(
	    action="python scripts/gpt_4o_image.py --prompt ${PROMPT} --input_directory ${INPUT} --output ${TARGETS[0]} --api_key ${API_KEY}"
	    ),
	"LLavaCorrect" : Builder(
	    action="python scripts/llava_correction.py  --image_directory ${INPUT} --prompt ${PROMPT} --output_file ${TARGETS[0]}"
	),

	"Compare_Performance" : Builder(
            action="python scripts/compare_performance_json.py --control_directory ${SOURCES[0]} --test_directory ${SOURCES[1]} --output_file ${TARGETS[0]}"
        ),

	"GenerateReport" : Builder(
            action="python scripts/condense_output.py --input_docs ${SOURCES} --output_file ${TARGETS[0]} --condensed_output ${TARGETS[1]} --control_file ${CONTROL}"
	    ),
	"AnalyzeOutput" : Builder(
	    action= "python scripts/analyze_error_rates.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        )
    }
)

# OK, at this point we have defined all the builders and variables, so it's
# time to specify the actual experimental process, which will involve
# running all combinations of datasets, folds, model types, and parameter values,
# collecting the build artifacts from applying the models to test data in a list.
#
# The basic pattern for invoking a build rule is:
#
#   "Rule(list_of_targets, list_of_sources, VARIABLE1=value, VARIABLE2=value...)"
#
# Note how variables are specified in each invocation, and their values used to fill
# in the build commands *and* determine output filenames.  It's a very flexible system,
# and there are ways to make it less verbose, but in this case explicit is better than
# implicit.
#
# Note also how the outputs ("targets") from earlier invocation are used as the inputs
# ("sources") to later ones, and how some outputs are also gathered into the "results"
# variable, so they can be summarized together after each experiment runs.

tesseract_results = []
for dataset in env["DATASETS"]:
    tesseract_results.append(env.Perform_OCR("work/pytesseract_ocr.json", INPUT = env["IMAGE_LOCATION"]))

initial_comparison = []
for tesseract_result in tesseract_results:
    initial_comparison.append(env.ComparePerformanceInitial("work/initial_comparison.json", tesseract_result, CONTROL= "/home/sbacker2/projects/post_ocr_correction/data/texts/"))

gpt_correct = []
print(env["GPT_VERSION"])
for model in env["GPT_VERSION"]:
    for prompt in env["PROMPTS"]:
    	for tess_result in tesseract_results:
    	    gpt_correct.append([env.GPTCorrect("work/gpt_correction_{}.json".format(model), tess_result,PROMPT = prompt[1], MODEL_TYPE = model, API_KEY = env["API_KEY"]), prompt, model,tess_result])

#print("gpt_correct_length")
#print(len(gpt_correct))



GPT_Compared = []
Compared_Namelist = []
for comparison in initial_comparison:
    for corrected_result in gpt_correct:
        print(corrected_result[2])
        print(corrected_result[1])
        print(corrected_result[3])
        GPT_Compared.append([env.Compare_Performance("work/compared_with_gpt{}{}.json".format(corrected_result[2],corrected_result[1][0]), [comparison, corrected_result[0]]),comparison, corrected_result])
        Compared_Namelist.append("work/compared_with_gpt{}{}.json".format(corrected_result[2],corrected_result[1][0]))

print("check to see this happened")
local_prompt = "Please return all of the text contained in this  document. DO NOT include any commentary, discussion, or description of the image besides whatever text is included in it. DO NOT add any words or phrases not present."

input = env["IMAGE_LOCATION"]


#print("PROMPTS:", env["PROMPTS"][0])
#print("IMAGE_LOCATION:", env["IMAGE_LOCATION"])



llava_correction_test = env.LLavaCorrect(target = "work/llava_correction.json", source = [], INPUT = input, PROMPT = local_prompt, API_KEY = env["API_KEY"])


llava_compared = env.Compare_Performance("work/compared_with_llava.json", [initial_comparison[0], llava_correction_test])
Compared_Namelist.append("work/compared_with_llava.json")

gpt_image_to_text = env.GPT4OCorrect("work/corrected_by_gpt40.json", source = [], INPUT = input, PROMPT = local_prompt)

gpt_4o_compared = env.Compare_Performance("work/compared_by_gpt40.json",[initial_comparison[0],gpt_image_to_text]) 
Compared_Namelist.append("work/compared_by_gpt40.json")

#print("this is comparison length")
#print(len(initial_comparison))
#print("this is namelist length")
#print(len(Compared_Namelist))
#print("this is namelist!")
print(Compared_Namelist)
#print(initial_comparison[0])


final_report = env.GenerateReport(["work/final_report.json","work/final_report_condensed.json"],Compared_Namelist,CONTROL=initial_comparison[0]) 

Analyzed_Output = env.AnalyzeOutput("work/analyzed_output_report.json", final_report[1])

# Use the list of applied model outputs to generate an evaluation report (table, plot,
# f-score, confusion matrix, whatever makes sense).

