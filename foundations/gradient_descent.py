class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        #gradient -way to teach a model by repeatedly makig small connections.
        #it tells us which direction changes the weights to reduce the loss
        #we nudge the weights a little in that direction
        #repeat many times
        #new_weight=old_weight-learning_rate*2;
        #loss=(prediction-target)^2
        #A common loss is a squared error
        #let say the model make prediction of 5 i,.e is old_weight=5(initial weight or init or w)
        #but model was made to make prediction =2,so this is target
        x=init
        for i in range(iterations):
            x=x-learning_rate*2*x
            
        return round(x,5)
        pass
