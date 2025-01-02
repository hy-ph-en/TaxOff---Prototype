# TAX-OFF

A safe and efficient tax optimization system leveraging LLMs to create personalized taxation plans for UK businesses and individuals. The system integrates UK governmental schemes and tax incentives to provide optimal financial planning while maintaining compliance and safety.

## Key Features
- LLM-powered tax optimization
- UK tax scheme integration  
- Safe system architecture
- Multi-disciplinary approach
## To Run:

### LOCAL [LINUX/MAC]
1. Clone the repository
2. cd to the repository
3. run `bash run.sh <openai-api-key>`
4. Open the browser and go to `http://localhost:8501/`


### DOCKER
1. Clone the repository
2. cd to the repository
3. run `docker build --build-arg OPENAI_API_KEY=<openai-api-key> -t tax-off:1 .`
4. run `docker run -p 8501:8501 tax-off:1`
5. Open the browser and go to `http://localhost:8501/`

