@echo off

set chatbotCmd=C:\Users\cn053\.conda\envs\ShuntianENV\python.exe chatbot.py

set agentCmd=C:\Users\cn053\.conda\envs\ShuntianENV\python.exe chatagent.py

set labCmd="C:\Users\cn053\.conda\envs\ISPython\python.exe" -m jupyter lab

set chainCmd=C:\Users\cn053\.conda\envs\ShuntianENV\python.exe serviceChain\TaskChainUI.py

set kmeansCmd="C:\Users\cn053\.conda\envs\ISPython\python.exe" -m streamlit run k-means.py --server.port 7875 --server.headless true

set forestCmd="C:\Users\cn053\.conda\envs\ISPython\python.exe" -m streamlit run random_forest.py --server.port 7880 --server.headless true

set mainCmd="C:\Users\cn053\.conda\envs\ISPython\python.exe" -m uvicorn main:app --port 8000 --host 0.0.0.0

set javaCmd=java -jar JavaBackEnd.jar

start "Chat bot Server" cmd /k %chatbotCmd%

start "Chat Agent Server" cmd /k %agentCmd%

start "Jupyter Lab Server" cmd /k %labCmd%

start "Chain Server" cmd /k %chainCmd%

start "K-means Server" cmd /k %kmeansCmd%

start "Random Forest Server" cmd /k %forestCmd%

start "Main Server" cmd /k %mainCmd%

start "Java Server" cmd /k %javaCmd%

exit