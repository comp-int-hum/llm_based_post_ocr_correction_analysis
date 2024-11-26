# Work on exploring LLM based vision-to-text + llm based post-ocr correction within the disciplinary conventions of historical research 

Sorry anyone reading this—the thing is currently a mess. I (sam) still need to pull everything into the build system, update requirements, etc. 

# Basic Question 

A) does llm-based correction work — and by work, we mean what is the error rate, and how does that impact down stream tasks for historians 

# what is this doing 

a) taking a series of gold-standard transcriptions from the Library of Congress's "We The People" Samuel Gompers project and comparing it to various ocr outputs 

those outputs include pytesseract, gpt3.5, gpt4 (both zero shot corrections) -- gpt 40 vision to text, bart (base, and then finetuned with both gold standard data, and silver quality data from gpt 40)

error rates are examined -- comparing traditional methods and more historically-specific questions 

B) NERs are identified, and compared between methods. The embeddings of these NERs are also compared, simulating potential downstream historical usage. 
