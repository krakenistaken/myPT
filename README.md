This agent makes a diet plan (in dietplan.txt) using the personal informations about user(personalinfo.txt). This agent is actually a part of a mobile app so we can make diet plans for users using the data they give us.

LLM model can be changed easily, To change the LLM, you need to add the new provider's package to requirements.txt, install it using "pip install -r requirements.txt", update the import statement in diet_agent.py, and replace the ChatOllama instantiation inside build_agent() with the new model class.

