import streamlit as st
from openai import OpenAI
import time
from datetime import datetime
import json
import statistics
import re

# Setup OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def generate_followup_question(learning_goal):
    prompt = f"""As a motivational coach, ask a concise, open-ended question to help a learner reflect on their reasons for pursuing "{learning_goal}". Keep the answer short."""
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528-qwen3-8b:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        content = response.choices[0].message.content.strip() if response.choices else ""
        return content if content else "What does this goal mean to you personally, and how do you want it to impact your future?"
    except Exception as e:
        return "What inspires your desire to pursue this goal, and what do you hope to achieve?"

def analyze_response_gemma(response_text, time_taken, session_num, goal, history=None):
    prompt = f"""
You are an expert language analyst. A learner wants to improve at: "{goal}".
Here is their reflective response:
"{response_text}"
Evaluate on a 0–1 scale:
1. Conceptual Vocabulary (depth, variety, originality)
2. Clarity of Expression
3. Emotional Tone Strength
Respond in JSON:
{{
  "conceptual_vocab_score": 0.0,
  "clarity_score": 0.0,
  "emotional_tone_score": 0.0
}}
Then write 1 sentence each explaining why:
- Conceptual Vocabulary: ...
- Clarity: ...
- Emotional Tone: ...
"""
    try:
        result = client.chat.completions.create(
            model="google/gemma-3n-e2b-it:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300
        )
        raw_output = result.choices[0].message.content.strip()
        json_block = re.search(r"\{.*?\}", raw_output, re.DOTALL)
        scores = json.loads(json_block.group(0)) if json_block else {
            "conceptual_vocab_score": 0.5,
            "clarity_score": 0.5,
            "emotional_tone_score": 0.5
        }
        explanations = {
            "conceptual_vocab_feedback": "Not found",
            "clarity_feedback": "Not found",
            "emotional_tone_feedback": "Not found"
        }
        for line in raw_output.splitlines():
            if line.lower().startswith("- conceptual"):
                explanations["conceptual_vocab_feedback"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("- clarity"):
                explanations["clarity_feedback"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("- emotional"):
                explanations["emotional_tone_feedback"] = line.split(":", 1)[1].strip()
    except Exception as e:
        scores = {
            "conceptual_vocab_score": 0.5,
            "clarity_score": 0.5,
            "emotional_tone_score": 0.5
        }
        explanations = {
            "conceptual_vocab_feedback": "Default due to error.",
            "clarity_feedback": "Default due to error.",
            "emotional_tone_feedback": "Default due to error."
        }
    latency_std = round(statistics.stdev([s["time_taken"] for s in history]), 2) if history and len(history) > 1 else None
    return {**scores, **explanations, "time_to_completion": time_taken, "latency_variability": latency_std}

def generate_practice_pointers(goal, response, scores=None):
    prompt = f"""
You are an adaptive learning coach helping a learner improve at: "{goal}".
Here is what they said:
"{response}"
Their current skill indicators (0–1 scale) are:
- Conceptual Vocabulary: {scores.get('conceptual_vocab_score', 0.0)}
- Clarity: {scores.get('clarity_score', 0.0)}
- Emotional Tone: {scores.get('emotional_tone_score', 0.0)}
Suggest 2–3 specific, doable next steps they can take to improve based on their current level. Be brief, direct, and realistic.
Respond ONLY with bullet points.
"""
    try:
        result = client.chat.completions.create(
            model="qwen/qwen3-235b-a22b:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        return "- Try writing a short journal about your thoughts.\n- Watch a simple video about your topic and reflect on what stands out."

# Streamlit app
st.title("🎯 Adaptive Learning System")
st.write("A personalized learning tool to help you reflect and improve.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.session_num = 1
    st.session_state.learning_goal = None
    st.session_state.followup_question = None
    st.session_state.start_time = None

# Input learning goal
if not st.session_state.learning_goal:
    learning_goal = st.text_input("🧠 What are you trying to learn today?", key="learning_goal")
    if learning_goal:
        st.session_state.learning_goal = learning_goal
        st.session_state.followup_question = generate_followup_question(learning_goal)
        st.rerun()

# Main session loop
if st.session_state.learning_goal:
    st.subheader(f"Session {st.session_state.session_num}")
    
    # Display reflection question
    if st.session_state.session_num == 1:
        st.write(f"🔍 Reflect on this: {st.session_state.followup_question}")
        user_response = st.text_area("✍️ Your response:", key=f"response_{st.session_state.session_num}")
        if user_response:
            st.session_state.start_time = time.time()
    else:
        progress_update = st.text_area(f"🔁 Reflect: What progress have you made toward '{st.session_state.learning_goal}' since last session?", 
                                      key=f"progress_{st.session_state.session_num}")
        if progress_update:
            progress_prompt = f"""
The learner's goal is: "{st.session_state.learning_goal}".
Their progress update is: "{progress_update}".
Generate ONE short, emotionally insightful follow-up reflection question. Keep it under 25 words. Output ONLY the question.
"""
            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                    messages=[{"role": "user", "content": progress_prompt}],
                    temperature=0.5,
                    max_tokens=50
                )
                followup_question = re.search(r"(.+?\?)", response.choices[0].message.content.strip())
                st.session_state.followup_question = followup_question.group(1).strip() if followup_question else "What did you feel during your practice?"
            except Exception as e:
                st.session_state.followup_question = "What did you feel during your practice?"
            
            st.write(f"🔍 Reflect on this: {st.session_state.followup_question}")
            user_response = st.text_area("✍️ Your response:", key=f"response_{st.session_state.session_num}")
            if user_response:
                st.session_state.start_time = time.time()

    # Process response
    if st.session_state.start_time and user_response:
        time_taken = round(time.time() - st.session_state.start_time, 2)
        timestamp = datetime.now().isoformat()
        
        analysis = analyze_response_gemma(
            response_text=user_response,
            time_taken=time_taken,
            session_num=st.session_state.session_num,
            goal=st.session_state.learning_goal,
            history=st.session_state.history
        )
        
        suggestions = generate_practice_pointers(
            goal=st.session_state.learning_goal,
            response=user_response,
            scores=analysis
        )
        
        # Display results
        st.write("✅ **SESSION SUMMARY**")
        st.write(f"📆 **Timestamp**: {timestamp}")
        st.write(f"🧠 **Goal**: {st.session_state.learning_goal}")
        st.write(f"🗣️ **Reflection**: {user_response}")
        
        st.write("📊 **SCORES**")
        for key in ["conceptual_vocab_score", "clarity_score", "emotional_tone_score"]:
            st.write(f"• {key.replace('_', ' ').title()}: {analysis[key]}")
        
        st.write("💬 **FEEDBACK**")
        for key in ["conceptual_vocab_feedback", "clarity_feedback", "emotional_tone_feedback"]:
            st.write(f"• {analysis[key]}")
        
        st.write("🎯 **NEXT STEPS**")
        st.markdown(suggestions)
        
        # Store session data
        st.session_state.history.append({
            "session_num": st.session_state.session_num,
            "learning_goal": st.session_state.learning_goal,
            "followup_question": st.session_state.followup_question,
            "response": user_response,
            "time_taken": time_taken,
            "timestamp": timestamp,
            **analysis,
            "practice_suggestions": suggestions,
        })
        
        # Option to continue
        if st.button("🔁 Do another session?"):
            st.session_state.session_num += 1
            st.session_state.start_time = None
            st.rerun()