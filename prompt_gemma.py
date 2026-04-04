import ollama
import requests

def search_literature(query: str, limit: int = 5):
    """
    Calls the Europe PMC REST API to search for literature based on the query.
    Returns a list of dictionaries containing title, abstract, and DOI.
    """
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "resultType": "core", # 'core' provides abstract text
        "pageSize": limit
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    results = []
    for item in data.get("resultList", {}).get("result", []):
        results.append({
            "title": item.get("title", "No Title"),
            "abstract": item.get("abstractText", "No abstract available"),
            "doi": item.get("doi", "No DOI available")
        })
    return results

# Native Gemma tool schema definition
search_literature_tool = {
    'type': 'function',
    'function': {
        'name': 'search_literature',
        'description': 'Use this tool to search Europe PMC for scientific literature when asked about drug-target interactions. Returns a list of article titles, abstracts, and DOIs.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The drug-target search query string (e.g., drug names, gene targets, interactions).'
                },
                'limit': {
                    'type': 'integer',
                    'description': 'The maximum number of literature results to return. Default is 5.'
                }
            },
            'required': ['query']
        }
    }
}

def main():
    model_name = 'gemma4:e2b'
    
    # System prompt to enforce structured citations in the final output
    system_prompt = (
        "You are a clinical research assistant. When summarizing literature, "
        "you MUST use in-line citations (e.g., [1], [2]) and include a "
        "'References' section at the bottom containing the Title and DOI "
        "of the papers you used."
    )

    # We switch to ollama.chat to use messages structure required for tool calling
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': 'What are the primary molecular mechanisms of acquired resistance to the PARP inhibitor Olaparib in BRCA-mutated cancers, and what alternative targets or combinations are being explored to overcome this?'}
    ]
    
    print(f"Prompting model '{model_name}' to see if it calls the tool...")
    
    try:
        # Pass the tool schema in the chat request
        response = ollama.chat(
            model=model_name,
            messages=messages,
            tools=[search_literature_tool]
        )
        
        message = response.get('message', {})
        
        # Check if the model opted to use the tool
        if 'tool_calls' in message and message['tool_calls']:
            print("\n=== Tool Call Triggered ===")
            for tool_call in message['tool_calls']:
                function_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                
                print(f"Function Name: {function_name}")
                print(f"Arguments Executed: {arguments}")
                
                if function_name == 'search_literature':
                    # Safely extract arguments
                    query = arguments.get('query')
                    limit = arguments.get('limit', 5)
                    
                    print(f"\nExecuting search_literature locally with query: '{query}'...")
                    literature_results = search_literature(query=query, limit=limit)
                    
                    # Format results with numbered references so the model can cite them
                    formatted_results = ""
                    for idx, paper in enumerate(literature_results, 1):
                        formatted_results += (
                            f"[{idx}] Title: {paper['title']}\n"
                            f"    DOI: {paper['doi']}\n"
                            f"    Abstract: {paper['abstract']}\n\n"
                        )
                    
                    # Adding the tool output back into the message history
                    messages.append(message) # the assistant's action message
                    messages.append({
                        'role': 'tool',
                        'content': f"Search results:\n\n{formatted_results}"
                    })
                    
                    print("\nFeeding the fetched literature back to the model for a final summary...")
                    final_response = ollama.chat(
                        model=model_name,
                        messages=messages,
                        tools=[search_literature_tool]
                    )
                    
                    print("\n=== Final Model Response ===")
                    print(final_response['message']['content'])
        else:
            print("\n=== Model Response (No tool called) ===")
            print(message.get('content', ''))
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
