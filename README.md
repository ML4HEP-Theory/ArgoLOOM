# ArgoLOOM, Quarks2Cosmos

Living directory containing the ArgoLOOM agentic codebase of the Quarks2Cosmos project. The core is an
agentic script which can invoke downstream toolkits relevant for cosmology, collider phenomenology, and DIS
experiments related to nuclear physics. Currently, the available base consists of the standard CLASS Boltzmannian
code for large-scale structure in the cosmic microwave background; MadGraph5 (MG5) to generate collider events; and
kinematical scripts to map high-energy collider events to corresponding DIS kinematics and predict detector effects
at the associated values of, e.g., (x,Q2).

This code set assumes a working installation of CLASS and MG5, to which the agent must be pointed after invoking
agent-chat-bot.py to initiate a dialogue with

   python agent-chat-bot.py .

It similarly invokes a knowledge base of technical references in the internal kb_out/ directory; the agent must also be
pointed toward the location of this knowledge base directory. Independent tools are also included (kb_build.py, kb_query.py)
to expand the knowledge base from .pdf files or arXiv identifiers as well as query the resulting FAISS indices.

ArgoLOOM is currently intended to run on local workstations and has been tested on Apple Silicon as well as Linux distributions
like Debian GNU/Linux 13 (trixie). It is recommended to unpack and build ArgoLOOM in an environment alongside MG5 and CLASS
with consistent python libraries as can be accomplished using conda.

ArgoLOOM at present also requires a number of standard python dependencies, including openai, numpy, pyarrow, matplotlib, pylhe,
faiss, and sentence-transformers; these can be installed via simple

   pip install openai ,

etc. Following the setup of necessary dependencies, users should set their OpenAI API key as an environment variable with

   export OPENAI_API_KEY="your_api_key_here"
   
as appropriate for Linux/macOS.

Comments or suggestions should be sent to Tim Hobbs (tim@anl.gov).
