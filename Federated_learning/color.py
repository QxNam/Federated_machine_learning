class clr():
    '''
    Custom color text
    '''
    def __init__(self):
        self.R = '\033[91m'
        self.G = '\033[92m'
        self.B = '\033[94m'
        self.Y = '\033[93m'
        self.BOLD = '\033[1m'
        self.E = '\033[0m'

    def ERROR(self, msg):
        return self.BOLD + self.R + msg + self.E
    
    def SUCCESS(self, msg):
        return self.BOLD + self.G + msg + self.E
    
    def WARNING(self, msg):
        return self.BOLD + self.Y + msg + self.E
    
    def TEXT(self, msg):
        return self.BOLD + self.B + msg + self.E
    
    def TEXT_BOLD(self, msg):
        return self.BOLD + msg + self.E
