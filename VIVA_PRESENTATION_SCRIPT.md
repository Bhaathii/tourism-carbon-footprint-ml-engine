# Tourism Carbon Footprint ML Engine - Viva Presentation Script (5 minutes)

---

## **INTRODUCTION (0:00 - 0:45 seconds)**

Good [morning/afternoon]. My name is [Your Name], and I'm presenting my final year project: **Tourism Carbon Footprint ML Engine**.

The problem we're solving is simple: tourists have no easy way to understand their environmental impact when traveling. Tourism is a major industry, but it generates significant carbon emissions through transportation, accommodation, food, and waste.

Our solution is a web-based application that helps tourists predict their carbon emissions and get personalized eco-friendly recommendations. The application uses a combination of **machine learning for predictions** and **Llama AI for smart recommendations**, making it both intelligent and user-friendly.

---

## **TECHNICAL STACK (0:45 - 1:45 seconds)**

Let me explain the key technologies we used:

### **Frontend - Streamlit**
We built the user interface using **Streamlit**, a Python framework designed specifically for data science and machine learning applications. Streamlit is perfect for us because:
- It's fast to develop with—no complex HTML or JavaScript needed
- It provides interactive widgets for user input
- It makes visualization simple and clean
- Most importantly, it lets data scientists focus on the logic, not web design

### **Machine Learning - Random Forest**
For the core prediction engine, we trained a **Random Forest Classifier** on 5,000+ tourism trip records. This model classifies carbon emissions into three levels: **Low, Medium, and High**. Random Forest works well here because it handles multiple features nicely and provides confidence scoring.

### **AI Recommendations - Ollama and Llama**
Now, here's the innovative part: after the ML model predicts the emission level, we pass that result to **Ollama**, which runs a local **Llama language model**. This is crucial because Llama generates **personalized, context-aware recommendations** based on:
- The emission level (low, medium, or high)
- The specific location in Sri Lanka the user is visiting
- The type of vehicle they're using

Llama understands natural language, so it can write human-friendly suggestions like "Take the scenic train from Colombo to Ella—it's 80% lower emissions" instead of just generic text.

### **Development Environment**
We developed this entirely in **VS Code**, using Python for all backend logic and Streamlit for the interactive dashboard.

---

## **WORKFLOW & DATA FLOW (1:45 - 3:15 seconds)**

Let me walk you through how the application works end-to-end:

**Step 1: User Input**
The user opens the Streamlit app and enters 13 different parameters across organized sections:
- Trip details (duration, distance, vehicle occupancy)
- Environmental factors (congestion, terrain)
- Accommodation information (electricity usage, grid efficiency)
- Food and waste data
- Transport and generator emissions

**Step 2: Input Validation**
Before anything else, we validate the inputs. For example:
- A Private Car can't hold 27 passengers
- Food emissions can't be negative
- Trip distance must be realistic

This prevents impossible scenarios from reaching the model.

**Step 3: ML Model Prediction**
The validated data flows into our Random Forest model, which:
- Processes all 13 features
- Returns a classification: Low, Medium, or High emission level
- Also gives us a confidence score

**Step 4: Llama AI Recommendation Generation**
Here's where Llama comes in. We send the emission level, trip details, and location context to Ollama, which runs the Llama model locally. Llama generates a detailed, personalized recommendation. For example:
- If the user is in Ella with High emissions, Llama might recommend: "Consider taking the train from Colombo to Ella—one of the world's most scenic train routes with 80% lower emissions than driving."
- The recommendations are location-specific because we feed Llama data about sustainable options in each Sri Lankan destination.

**Step 5: Results Display**
The app displays:
- A color-coded emission level (green for Low, yellow for Medium, red for High)
- A confidence score from the ML model
- A personalized recommendation from Llama
- Actionable next steps

All of this happens in real-time—under 2 seconds on average.

---

## **KEY FEATURES (3:15 - 4:15 seconds)**

Our application has three standout features:

### **Feature 1: Comprehensive Input Handling**
We collect 13 detailed parameters covering every aspect of a trip. This isn't just distance × emissions; we factor in vehicle type, terrain difficulty, grid efficiency, food sources, and waste—giving truly accurate predictions.

### **Feature 2: AI-Powered Personalization via Llama**
Instead of generic recommendations, Llama generates **context-aware suggestions** that mention local solutions. For Colombo, it suggests e-tuks and carpooling apps. For Galle, it recommends the coastal train. For Yala, it suggests shared safari jeeps. This is because Llama understands the specific geography and sustainable options of Sri Lanka.

### **Feature 3: Multi-Layer Intelligence**
The application doesn't rely on a single model. It combines:
- Statistical validation (to catch input errors)
- Machine Learning (for emission classification)
- Large Language Models (for human-friendly recommendations)

This makes it robust and trustworthy.

---

## **RESULTS & EVALUATION (4:15 - 5:00 seconds)**

Our model was trained on 5,000 tourism trip records and tested on a separate set:

- **Accuracy: 94.2%** — The model correctly classifies emission levels in nearly 95 out of 100 cases
- **Balanced Performance** — The model performs well across all three classes (Low, Medium, High)
- **Response Time: < 2 seconds** — Fast enough for real-time web use

For Llama recommendations, we validated them manually by checking:
- Whether suggestions are relevant to the location
- Whether they're factually accurate (e.g., the Ella train route actually exists and is efficient)
- Whether they provide actionable steps, not vague advice

The combination of **high-accuracy ML predictions** and **sensible AI recommendations** makes this a practical tool tourists and travel agencies can actually use.

---

## **CONCLUSION (5:00 - 5:03 seconds)**

In summary, we've built a tourism carbon calculator that combines machine learning for accuracy with Llama AI for intelligence. The result is an easy-to-use web app that helps tourists understand and reduce their environmental impact—one trip at a time.

Thank you, and I'm ready for your questions.

---

# **5 POTENTIAL VIVA QUESTIONS & ANSWERS**

---

## **Question 1: Why did you choose Ollama and Llama instead of a cloud-based API like OpenAI's GPT-4?**

### **Answer:**
Great question. There are three reasons:

1. **Privacy**: Ollama runs locally on the machine. User data never leaves the system, which is critical for a tourism calculator that might contain personal travel details.

2. **Cost**: Cloud APIs charge per request. With Ollama, we run the model once on local hardware, so there's no recurring API cost.

3. **Control**: We can control which model runs, how long it takes, and what it outputs. With a cloud API, we're at the mercy of their service availability and rate limits. For a Viva project, local deployment is much more reliable.

Additionally, Llama is an open-source model trained on diverse data, so it understands general knowledge about Sri Lankan locations and sustainable travel—which is exactly what we need here.

---

## **Question 2: How does the data flow from the Streamlit UI to the Llama model, and can you explain what happens at each step?**

### **Answer:**
Sure. Here's the exact flow:

1. **Streamlit collects input**: The user fills in 13 fields across 7 organized sections (Trip Details, Environmental Factors, Accommodation, Food & Waste, Transport, etc.).

2. **Input validation layer**: Before anything else, we validate the data. For example, if someone enters a Private Car with 25 passengers, our validation catches this and rejects it.

3. **ML model prediction**: The validated data is converted into a Pandas DataFrame with features in the exact order the Random Forest expects. The model returns: `emission_level` (low/medium/high) and `confidence_score`.

4. **Prompt construction for Llama**: We build a text prompt that includes:
   - The emission level from the ML model
   - The specific location (e.g., "Ella")
   - The vehicle type (e.g., "Private Car")
   - Relevant context (e.g., "Ella is known for its train route...")

5. **Ollama API call**: We call Ollama locally with this prompt and Llama generates a few paragraphs of personalized recommendations.

6. **Output parsing**: We parse Llama's response to clean it up and ensure it doesn't contain hallucinated numbers.

7. **Display results**: Streamlit renders the emission level badge, confidence score, and Llama's recommendation to the user.

The entire flow takes under 2 seconds.

---

## **Question 3: How does Llama know about specific locations in Sri Lanka, and could it hallucinate incorrect recommendations?**

### **Answer:**
Excellent point. This is something we explicitly handled in two ways:

1. **Location Context Injection**: Before sending data to Llama, we don't just say "location: Ella." Instead, we inject our own context. For example, for Ella, we include: "The Ella Odyssey train is one of the world's most scenic rail journeys and has a carbon footprint 80% lower than a private car. The town has several vegan-friendly rice and curry spots with local, low-emission produce."

   By providing factual context upfront, we "ground" Llama's responses in reality.

2. **Hallucination Detection**: After Llama generates recommendations, we run a post-processing step that:
   - Detects if Llama made up any false numbers (e.g., claiming a 99% emission reduction when that's unrealistic)
   - Sanitizes the output to remove potentially false claims
   - Falls back to pre-written, curated recommendations if Llama hallucinates

So while Llama is powerful at natural language, we don't blindly trust it. We combine it with our domain knowledge to ensure accuracy.

---

## **Question 4: What's the difference between the Random Forest model's prediction and Llama's role? Why do you need both?**

### **Answer:**
Great question—this is a common confusion point.

**Random Forest's job:**
- Takes 13 numerical/categorical features
- Produces a statistical classification: Low/Medium/High
- Gives us confidence in the prediction
- Fast and deterministic (same inputs = same output every time)
- But it's a "black box"—it doesn't explain *why* the emissions are high

**Llama's job:**
- Takes the classification output and generates *explanations and suggestions*
- Turns a statistical prediction into human-readable insights
- Makes recommendations location-specific and actionable
- But it's not good at numerical calculations—it can hallucinate numbers

**Why both?**
- Random Forest is accurate but not explanatory
- Llama is explanatory but can be unreliable with numbers

Together, they're more powerful: Random Forest gives us *accuracy*, and Llama gives us *interpretability and actionability*. A user cares less about the math and more about: "What should I do about my emissions?" Llama answers that question in a way Random Forest alone cannot.

---

## **Question 5: How would you handle a scenario where a user's internet connection fails while using the app? Does Ollama/Llama require the internet?**

### **Answer:**
This is an important practical question.

**The good news:** Ollama runs **locally**, not on the internet. Once Ollama and the Llama model are installed and running on the user's machine, the app works offline—no internet needed.

**The workflow:**
1. User installs Ollama and pulls the Llama model (this requires internet, but only once)
2. Ollama service runs on the local machine (usually on `localhost:11434`)
3. Streamlit connects to this local service
4. The app works entirely offline after that

**What happens if Ollama crashes or is unavailable:**
We have a fallback mechanism. If the Ollama API doesn't respond, the app automatically falls back to pre-written, curated recommendations specific to each location and emission level. So the user still gets a helpful response—just a static, pre-written one instead of a dynamically generated one.

This is a huge advantage over cloud-based AI services: reliability and offline capability.

---

## **BONUS: Why did you train the model on tourism data specifically?**

[If asked]

### **Answer:**
Tourism is a specific use case with unique factors. A regular carbon calculator might focus on daily commuting (distance × fuel type), but tourism involves:
- Varying trip durations (3 days to 3 weeks)
- Accommodation impact (hotels use significantly more electricity than homes)
- Food variability (tourists often eat differently than they do at home—more restaurant meals, often imported food)
- Vehicle sharing (tourists often share vans, unlike daily commuters)

By training on 5,000 real tourism trip records, the Random Forest learned the actual distribution of these factors and their weights. The model discovered that, for example, accommodation and food often contribute more to a tourist's footprint than a daily commuter would expect. This makes our predictions specifically tailored to tourism, not generic.

---

**Good luck with your Viva! 🎓**
