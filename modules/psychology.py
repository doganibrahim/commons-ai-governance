"""
Ref: Ostrom's Principles, Self-Determination Theory, and Equity Theory.
"""

import math

class PsychologyModel:
    def __init__(self):
        pass

    @staticmethod
    def calculate_initial_trust(base_trust=50, is_ai_system=False, penalty=15):
        """
        Calculates initial trust based on the system type.
        Formula: Trust_Initial = G_baseline - (G_penalty if AI)
        """
        if is_ai_system:
            return max(0, base_trust - penalty)
        return base_trust

    @staticmethod
    def update_trust(current_trust, experience_type):
        """
        Updates trust based on interaction experience.
        
        Args:
            current_trust (float): 0-100
            experience_type (str): 'positive', 'negative'
        """
        if experience_type == 'positive':
            delta = 100 / (10 + current_trust)
            return min(100, current_trust + delta)
        
        elif experience_type == 'negative':
            # Trust is lost much faster (Glass Cannon effect)
            delta_negative = max(10, current_trust * 0.4)

            return max(0, current_trust - delta_negative)
            
        return current_trust

    @staticmethod
    def calculate_satisfaction(inputs_x, outputs_y, weights_v, weights_u, procedural_bonus=0):
        """
        Calculates satisfaction using simplified DEA logic + Procedural Justice Bonus.
        
        Formula: Satisfaction = (Weighted Output / Weighted Input) + Bonus
        Note: Real DEA requires linear programming. Here we calculate the ratio 
        based on agent's subjective weights.
        """

        weighted_input = max(0.1, inputs_x * weights_v)
        weighted_output = outputs_y * weights_u
        
        ratio = weighted_output / weighted_input
        
        # Simple normalization: If ratio >= 1 (fair), score is high.
        # This approximates the 'Efficiency Score' in DEA
        dea_score = min(1.0, ratio) 
        
        # Add Procedural Justice Bonus
        final_score = min(1.0, dea_score + procedural_bonus)
        
        return final_score

    @staticmethod
    def calculate_cooperation_cost(base_cost, autonomy_felt, k=1.0):
        """
        Calculates the subjective cost of cooperation based on autonomy.
        
        Formula: Cost_Effective = Cost_Base / (1 + k * Autonomy)
        """
        denominator = 1 + (k * autonomy_felt)
        return base_cost / denominator

    @staticmethod
    def calculate_cooperation_probability(trust, satisfaction, autonomy, scarcity, weights):
        """
        Calculates the probability of cooperation.
        
        Formula: f(w1 * Trust, w2 * Satisfaction, w3 * Autonomy, w4 * Scarcity)
        """
        w1, w2, w3, w4 = weights
        
        # Normalize trust to 0-1 for consistent calculation
        norm_trust = trust / 100.0
        
        # Score = w1*T + w2*S + w3*A - w4*Scarcity (Scarcity increases competition)
        
        score = (w1 * norm_trust) + \
                (w2 * satisfaction) + \
                (w3 * autonomy) - \
                (w4 * scarcity)
        
        # Convert score to probability using Sigmoid function
        probability = 1 / (1 + math.exp(-5 * (score - 0.5)))
        
        return min(1.0, max(0.0, probability))

    @staticmethod
    def update_autonomy(current_autonomy, event_type):
        """
        Updates autonomy level based on system events.
        """
        delta = 0.0
        if event_type == 'meaningful_choice':
            delta = 0.2
        elif event_type == 'justified_instruction':
            delta = 0.1
        elif event_type == 'conditional_reward':
            delta = -0.15
        elif event_type == 'unjustified_instruction':
            delta = -0.2
        elif event_type == 'forced_decision':
            delta = -0.2

        return max(0.0, min(1.0, current_autonomy + delta))