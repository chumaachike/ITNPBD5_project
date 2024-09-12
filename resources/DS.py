class DS:
    def __init__(self, size):
        self.data = [] #initialize an array to store a collection of objects
        self.size = size # inititialize the maximum size of the collection 

    def add(self, value, objective):
        # if the length of the array is less than the defined size, add value
        if len(self.data) < self.size:
            self.data.append(value)
        else:
            #if the array is full find the minimum or maximum depending on the objective
            if objective == "max":
                min_f_value_dict = min(self.data, key=lambda x: x['f'])
                if value['f'] > min_f_value_dict['f']:
                    self.data.remove(min_f_value_dict)
                    self.data.append(value)
            elif objective == "min":
                max_f_value_dict = max(self.data, key=lambda x: x['f'])
                if value['f'] < max_f_value_dict['f']:
                    self.data.remove(max_f_value_dict)
                    self.data.append(value)

    def get_data(self):
        # Return the entire list of dictionaries
        return self.data