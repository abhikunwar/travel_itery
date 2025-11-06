from langgraph.graph import StateGraph, END
from typing import Dict, List, TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI  # or any LLM
import operator
import json

# Define the state structure
class TravelState(TypedDict):
    destination: str
    duration: int
    budget: float
    user_preferences: Dict
    transport_options: Annotated[List, operator.add]
    accommodation_options: Annotated[List, operator.add]
    activity_options: Annotated[List, operator.add]
    weather_data: Dict
    itinerary: Dict
    messages: Annotated[List, operator.add]
    current_step: str

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


# Define Agents

class TravelAgents:
    @staticmethod
    def orchestrator_agent(state: TravelState):
        """Main agent that coordinates the planning process"""
        system_msg = """You are a travel planning orchestrator. Analyze the user request and 
        determine which specialized agents need to be involved. Your goal is to create a 
        comprehensive travel plan."""
        
        user_msg = f"""
        User Request:
        - Destination: {state['destination']}
        - Duration: {state['duration']} days
        - Budget: ₹{state['budget']}
        - Preferences: {state.get('user_preferences', {})}
        
        Determine the next steps and coordinate with specialized agents.
        """
        
        response = llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ])
        
        state['messages'].append(HumanMessage(content=user_msg))
        state['messages'].append(response)
        state['current_step'] = 'orchestrator'
        
        return state

    @staticmethod
    def budget_agent(state: TravelState):
        """Handle budget allocation and constraints"""
        budget_rules = """
        Budget Allocation Rules:
        - Accommodation: 40% of total budget
        - Transport: 30% of total budget  
        - Activities & Food: 30% of total budget
        - Always prioritize staying within budget
        """
        
        prompt = f"""
        {budget_rules}
        
        Total Budget: ₹{state['budget']} for {state['duration']} days
        Destination: {state['destination']}
        
        Provide a detailed budget allocation and recommendations.
        """
        
        response = llm.invoke([
            SystemMessage(content="You are a expert travel budget planner."),
            HumanMessage(content=prompt)
        ])
        
        # Parse and store budget allocation
        budget_data = {
            'total_budget': state['budget'],
            'allocation': {
                'accommodation': state['budget'] * 0.4,
                'transport': state['budget'] * 0.3,
                'activities': state['budget'] * 0.3
            },
            'recommendations': response.content
        }
        
        state['messages'].append(response)
        state['current_step'] = 'budget_planning'
        
        return state

    @staticmethod
    def weather_agent(state: TravelState):
        """Provide weather-based recommendations"""
        # Mock weather data -we will call weather API in real time.....
        weather_data = {
            'goa': {'summer': 'hot and humid', 'monsoon': 'heavy rains', 'winter': 'pleasant'},
            'manali': {'summer': 'cool', 'monsoon': 'moderate rains', 'winter': 'snowy'}
        }
        
        destination = state['destination'].lower()
        season = "summer"  # This would be determined from dates
        
        weather_info = weather_data.get(destination, {})
        
        prompt = f"""
        Destination: {state['destination']}
        Season: {season}
        Weather Conditions: {weather_info.get(season, 'Unknown')}
        
        Provide weather-appropriate activity recommendations and packing suggestions.
        """
        
        response = llm.invoke([
            SystemMessage(content="You are a travel weather expert."),
            HumanMessage(content=prompt)
        ])
        
        state['weather_data'] = {
            'destination': destination,
            'season': season,
            'conditions': weather_info.get(season, 'Unknown'),
            'recommendations': response.content
        }
        state['messages'].append(response)
        state['current_step'] = 'weather_analysis'
        
        return state

    @staticmethod
    def transport_agent(state: TravelState):
        """Handle transportation planning"""
        transport_options = [
            {'type': 'flight', 'cost': 6000, 'duration': '2h', 'class': 'economy'},
            {'type': 'train', 'cost': 3000, 'duration': '12h', 'class': 'sleeper'},
            {'type': 'bus', 'cost': 2000, 'duration': '16h', 'class': 'sleeper'},
            {'type': 'car', 'cost': 4000, 'duration': '10h', 'class': 'self-drive'}
        ]
        
        # Filter by budget
        transport_budget = state['budget'] * 0.3
        affordable_transport = [t for t in transport_options if t['cost'] <= transport_budget]
        
        prompt = f"""
        Destination: {state['destination']}
        Transport Budget: ₹{transport_budget}
        
        Available Options: {affordable_transport}
        
        Recommend the best transport options considering budget, comfort, and time.
        """
        
        response = llm.invoke([
            SystemMessage(content="You are a transportation planning expert."),
            HumanMessage(content=prompt)
        ])
        
        state['transport_options'] = affordable_transport
        state['messages'].append(response)
        state['current_step'] = 'transport_planning'
        
        return state

    @staticmethod
    def accommodation_agent(state: TravelState):
        """Handle accommodation planning"""
        accommodation_options = [
            {'name': 'Beach Guesthouse', 'cost_per_night': 1500, 'type': 'budget', 'rating': 4.0},
            {'name': 'Mid-range Hotel', 'cost_per_night': 3000, 'type': 'comfort', 'rating': 4.3},
            {'name': 'Boutique Resort', 'cost_per_night': 5000, 'type': 'luxury', 'rating': 4.7},
            {'name': 'Luxury Hotel', 'cost_per_night': 8000, 'type': 'premium', 'rating': 4.9}
        ]
        
        accommodation_budget = (state['budget'] * 0.4) / state['duration']
        affordable_accommodation = [a for a in accommodation_options 
                                  if a['cost_per_night'] <= accommodation_budget]
        
        prompt = f"""
        Destination: {state['destination']}
        Duration: {state['duration']} days
        Accommodation Budget: ₹{accommodation_budget} per night
        
        Available Options: {affordable_accommodation}
        
        Recommend the best accommodation considering budget, comfort, and user preferences.
        """
        
        response = llm.invoke([
            SystemMessage(content="You are an accommodation expert."),
            HumanMessage(content=prompt)
        ])
        
        state['accommodation_options'] = affordable_accommodation
        state['messages'].append(response)
        state['current_step'] = 'accommodation_planning'
        
        return state

    @staticmethod
    def itinerary_synthesizer(state: TravelState):
        """Synthesize all information into final itinerary"""
        prompt = f"""
        Synthesize all the collected information into a comprehensive travel itinerary:
        
        Destination: {state['destination']}
        Duration: {state['duration']} days
        Budget: ₹{state['budget']}
        
        Weather Considerations: {state.get('weather_data', {})}
        Transport Options: {state.get('transport_options', [])}
        Accommodation Options: {state.get('accommodation_options', [])}
        
        Create a detailed day-by-day itinerary with:
        1. Transportation details
        2. Accommodation booking
        3. Daily activities
        4. Budget breakdown
        5. Weather-appropriate recommendations
        """
        
        response = llm.invoke([
            SystemMessage(content="You are an expert travel itinerary planner. Create comprehensive, practical itineraries."),
            HumanMessage(content=prompt)
        ])
        
        state['itinerary'] = {
            'destination': state['destination'],
            'duration': state['duration'],
            'budget': state['budget'],
            'plan': response.content,
            'transport_selection': state.get('transport_options', [])[0] if state.get('transport_options') else None,
            'accommodation_selection': state.get('accommodation_options', [])[0] if state.get('accommodation_options') else None
        }
        
        state['messages'].append(response)
        state['current_step'] = 'itinerary_synthesis'
        
        return state
    
def build_travel_agent():
    """Build the complete travel agent workflow using LangGraph"""
    
    # Create the graph
    workflow = StateGraph(TravelState)
    
    # Add nodes (agents)
    workflow.add_node("orchestrator", TravelAgents.orchestrator_agent)
    workflow.add_node("budget_planner", TravelAgents.budget_agent)
    workflow.add_node("weather_advisor", TravelAgents.weather_agent)
    workflow.add_node("transport_planner", TravelAgents.transport_agent)
    workflow.add_node("accommodation_planner", TravelAgents.accommodation_agent)
    workflow.add_node("itinerary_synthesizer", TravelAgents.itinerary_synthesizer)
    
    # Define the workflow
    workflow.set_entry_point("orchestrator")
    
    # From orchestrator, parallel execution of specialized agents
    workflow.add_edge("orchestrator", "budget_planner")
    workflow.add_edge("budget_planner", "weather_advisor")
    workflow.add_edge("weather_advisor", "transport_planner")
    workflow.add_edge("transport_planner", "accommodation_planner")
    workflow.add_edge("accommodation_planner", "itinerary_synthesizer")
    workflow.add_edge("itinerary_synthesizer", END)
    
    # Compile the graph
    return workflow.compile()

# Build the agent
travel_agent = build_travel_agent()    