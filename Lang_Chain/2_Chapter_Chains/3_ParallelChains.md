This file introduces a new LangChain concept:

## Parallel Chains

![alt text](image-1.png)

![alt text](image-2.png)

Earlier you learned a linear chain:

```
Input → Prompt → LLM → Parser → Output
```

This file introduces branching pipelines:

```
Input
↓
LLM Summary
↓
Convert to dictionary
↓
Run two chains simultaneously
```

- LinkedIn post generator
- Instagram post generator

So the system generates two different outputs from the same input.

## Conceptual Architecture

The pipeline looks like this:

```
User Input
↓
Movie Summary Chain
↓
Dictionary Conversion
↓
Parallel Execution
```

- Branch 1 → LinkedIn Post
- Branch 2 → Instagram Post

```
↓
Combine Results
↓
Beautify Output
```

## Step-by-Step Explanation

### Step 1 — Load Environment Variables

**Purpose:**

Load API credentials from .env.

**Example .env values:**

```
OPENAI_API_KEY=xxxx
```

For Vertex AI we will instead use:

```
GOOGLE_CLOUD_PROJECT
GOOGLE_CLOUD_REGION
GEMINI_MODEL
```

This ensures configuration is external to the code.

### Step 2 — Create the First Prompt

This prompt asks the model to summarize the movie.

**Example input:**

```
KGF
```

**Prompt sent to LLM:**

```
System: You are a movie summarizer
Human: Please summarize the movie in brief: KGF
```

**Output might be:**

```
Rocky rises from poverty and infiltrates the Kolar Gold Fields...
```

### Step 3 — Initialize LLM

The LLM processes prompts and generates responses.

In the OpenAI version this was:

```
ChatOpenAI
```

In the Vertex version we will use:

```
ChatVertexAI
```

### Step 4 — Output Parser

LLMs return objects like:

```
AIMessage(content="...")
```

The parser extracts the plain text.

### Step 5 — Custom Runnable

This converts the LLM response into a dictionary.

**Example conversion:**

```
"The capital of France is Paris"
```

→

```
{"text": "The capital of France is Paris"}
```

**Why?**

Because the next prompts expect a variable named {text}.

### Step 6 — LinkedIn Chain

This chain generates a LinkedIn post.

**Input:**

```
{"text": "summary"}
```

**Prompt:**

```
System: You are a LinkedIn post generator
Human: Create a post for LinkedIn: {text}
```

**Output:**

```
Professional LinkedIn style content.
```

### Step 7 — Instagram Chain

This chain generates an Instagram caption.

Instead of defining it statically, it is wrapped in a function.

The function builds:

```
Prompt → LLM → Parser
```

and runs it.

The result is returned.

### Step 8 — RunnableParallel

![alt text](image-3.png)

![alt text](image-4.png)

This is the core concept of this file.

**RunnableParallel** runs multiple chains simultaneously.

```
RunnableParallel(
    branches={
        "linkedin": chain_linkedin,
        "instagram": insta_chain_runnable
    }
)
```

Both chains run at the same time.

**Output structure becomes:**

```
{
  "branches": {
      "linkedin": "...",
      "instagram": "..."
  }
}
```

### Step 9 — Final Chain

The final chain combines all steps.

**Pipeline:**

```
Movie Input
↓
Summarize Movie
↓
Convert to dictionary
↓
Parallel Chains (LinkedIn + Instagram)
```

### Step 10 — Beautify Runnable

This function simplifies the result.

**Original output:**

```
{
  "branches": {
      "linkedin": "...",
      "instagram": "..."
  }
}
```

**Beautified output:**

```
{
 "linkedin": "...",
 "instagram": "..."
}
```