"""
Personal Trainer Diet Agent
============================
A LangChain agent powered by Ollama (Llama 3) that reads personalInfo.txt
and generates a fully personalized diet plan like a real personal trainer.
Runs 100% locally — no API key needed.
"""

import os
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

# Path to personalInfo.txt (same directory as this script)
INFO_FILE = Path(__file__).parent / "personalInfo.txt"


# ─────────────────────────────────────────────
# Tools the agent can use
# ─────────────────────────────────────────────
@tool
def read_personal_info() -> str:
    """Read the user's personal information from personalInfo.txt.
    Returns the full contents of the file including body stats,
    fitness goals, dietary preferences, budget, and schedule."""
    if not INFO_FILE.exists():
        return "ERROR: personalInfo.txt not found. Please create it in the project directory."
    return INFO_FILE.read_text(encoding="utf-8")


@tool
def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> str:
    """Calculate Basal Metabolic Rate using the Mifflin-St Jeor equation.
    Args:
        weight_kg: Body weight in kilograms
        height_cm: Height in centimeters
        age: Age in years
        gender: 'male' or 'female'
    """
    if gender.lower() == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    return (
        f"BMR (Mifflin-St Jeor): {bmr:.0f} kcal/day\n"
        f"Sedentary TDEE (x1.2): {bmr * 1.2:.0f} kcal\n"
        f"Lightly active TDEE (x1.375): {bmr * 1.375:.0f} kcal\n"
        f"Moderately active TDEE (x1.55): {bmr * 1.55:.0f} kcal\n"
        f"Very active TDEE (x1.725): {bmr * 1.725:.0f} kcal"
    )


@tool
def calculate_macros(
    calories: float, protein_grams: float, fat_percentage: float = 25.0
) -> str:
    """Calculate macro split given total calories and desired protein.
    Args:
        calories: Total daily calorie target
        protein_grams: Desired protein intake in grams
        fat_percentage: Percentage of calories from fat (default 25%)
    """
    protein_cal = protein_grams * 4
    fat_cal = calories * (fat_percentage / 100)
    fat_grams = fat_cal / 9
    carb_cal = calories - protein_cal - fat_cal
    carb_grams = carb_cal / 4

    return (
        f"=== MACRO BREAKDOWN ===\n"
        f"Protein : {protein_grams:.0f}g  ({protein_cal:.0f} kcal | {protein_cal/calories*100:.0f}%)\n"
        f"Fat     : {fat_grams:.0f}g  ({fat_cal:.0f} kcal | {fat_percentage:.0f}%)\n"
        f"Carbs   : {carb_grams:.0f}g  ({carb_cal:.0f} kcal | {carb_cal/calories*100:.0f}%)\n"
        f"Total   : {calories:.0f} kcal"
    )


@tool
def calculate_per_meal_macros(
    total_protein: float,
    total_carbs: float,
    total_fat: float,
    total_calories: float,
    num_meals: int,
) -> str:
    """Split daily macros evenly across meals.
    Args:
        total_protein: Total daily protein in grams
        total_carbs: Total daily carbs in grams
        total_fat: Total daily fat in grams
        total_calories: Total daily calories
        num_meals: Number of meals per day
    """
    lines = [f"=== PER-MEAL TARGETS ({num_meals} meals/day) ==="]
    for i in range(1, num_meals + 1):
        lines.append(
            f"Meal {i}: ~{total_protein/num_meals:.0f}g protein | "
            f"~{total_carbs/num_meals:.0f}g carbs | "
            f"~{total_fat/num_meals:.0f}g fat | "
            f"~{total_calories/num_meals:.0f} kcal"
        )
    return "\n".join(lines)


@tool
def save_diet_plan(plan_text: str, filename: str = "diet_plan.txt") -> str:
    """Save the generated diet plan to a text file.
    Args:
        plan_text: The complete diet plan text to save
        filename: Output filename (default: diet_plan.txt)
    """
    output_path = Path(__file__).parent / filename
    output_path.write_text(plan_text, encoding="utf-8")
    return f"Diet plan saved to {output_path}"


# ─────────────────────────────────────────────
# Agent setup
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an elite personal trainer and sports nutritionist with 15+ years of experience.
Your name is Coach. You are direct, motivating, and deeply knowledgeable.

YOUR PROCESS:
1. FIRST: Use the read_personal_info tool to get ALL the client's data.
2. SECOND: Use calculate_bmr to compute their metabolic rate from the data you read.
3. THIRD: Use calculate_macros to determine the optimal macro split based on their
   calorie target and desired protein.
4. FOURTH: Use calculate_per_meal_macros to split the macros across their meals.
5. FIFTH: Create a COMPLETE 7-day diet plan that:
   - Stays within the budget per meal constraint
   - Hits the protein and calorie targets
   - Respects cuisine preferences and dislikes
   - Accounts for their workout schedule (more carbs around training)
   - Includes simple, practical meals (check cooking time preferences)
   - Specifies exact portions in grams
   - Includes a grocery list with estimated costs
6. SIXTH: Use save_diet_plan to save the complete plan to a file.

IMPORTANT RULES:
- ALWAYS respect the budget constraint. Suggest affordable protein sources.
- ALWAYS hit the protein target. Use whey protein if the client allows it.
- Keep meals simple and practical based on cooking time preferences.
- Match the cuisine preferences mentioned in the file.
- Place higher-carb meals around workout times.
- Include specific portion sizes (in grams) for every item.
- Add a weekly grocery list with estimated costs in the client's currency.
- Be motivating but realistic. No bro-science, only evidence-based advice.

FORMAT the diet plan clearly with:
- Daily overview with macro totals
- Each meal with exact portions
- Post-workout nutrition guidance
- Weekly grocery list with costs
- Supplement recommendations if applicable
- Brief tips for meal prep and adherence"""


def build_agent():
    """Build and return the diet planning agent using LangGraph."""
    llm = ChatOllama(
        model="llama3.2",
        temperature=0.3,
    )

    tools = [
        read_personal_info,
        calculate_bmr,
        calculate_macros,
        calculate_per_meal_macros,
        save_diet_plan,
    ]

    # create_react_agent from langgraph handles the tool-calling loop automatically
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  COACH — Your AI Personal Trainer & Dietitian")
    print("  Powered by Llama 3.2 (Ollama) — 100% Local")
    print("=" * 60)
    print()

    agent = build_agent()

    # Primary request: generate the full diet plan
    print("Generating your personalized diet plan...\n")

    result = agent.invoke({
        "messages": [
            HumanMessage(content=(
                "Read my personal information file and create a complete, detailed "
                "7-day diet plan for me. Make sure every meal stays within my budget, "
                "hits my protein targets, and matches my cuisine preferences. "
                "Calculate my BMR and macros first, then build the plan. "
                "Save the final plan to a file when you're done."
            ))
        ]
    })

    # Extract the final AI response
    final_message = result["messages"][-1]
    print("\n" + "=" * 60)
    print("  YOUR PERSONALIZED DIET PLAN")
    print("=" * 60)
    print(final_message.content)

    # Interactive follow-up loop
    print("\n" + "-" * 60)
    print("Ask Coach anything (type 'quit' to exit):")
    print("-" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nStay consistent, stay disciplined. See you next time!")
            break
        if not user_input:
            continue

        result = agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        final_message = result["messages"][-1]
        print(f"\nCoach: {final_message.content}")


if __name__ == "__main__":
    main()
