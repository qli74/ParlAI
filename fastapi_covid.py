import logging
import subprocess
import fastapi
THRE=15 #threshold of scores


app = fastapi.FastAPI()
logger = logging.getLogger("api")

##########################################################
# Launch a command with pipes
p = subprocess.Popen(['python -m parlai.scripts.interactive -mf models/covid7 --single-turn True'], shell=True,
                     stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)

# Wait for the parlai CLI to initialize
while True:
    line = p.stdout.readline()
    line = line.strip().decode()
    if line == '[  polyencoder_type: codes ]':
        logger.info("parlai ready for commands")
        break

@app.get("/")
def root(question: str = "What is mask?"):
    logger.info("Question: {}".format(question))
    # Send the question and get the output
    p.stdin.write(bytes(question, 'utf-8'))
    p.stdin.write(bytes("\n", 'utf-8'))
    p.stdin.flush()
    line=''
    while '[Polyencoder]' not in line: # Exclude warnings and other messages
        line = p.stdout.readline()
        line = line.strip().decode()  # To interpret as text, decode
    result = line.split('[Polyencoder]:')
    #print(result)
    result =result[-1].split('|')
    score=result[0]
    if float(score)>THRE:
        answer=result[1]
    else: #skip if score<15
        answer="Sorry, I don't know."
    logger.info("Answer: {}".format(answer))
    logger.info("Score: {}".format(score))
    return {"question": question, "answer": answer,"score":score}

