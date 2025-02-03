from datetime import datetime


class bcolors:
    HEADER     = "\033[95m"
    OKBLUE     = "\033[94m"
    OKGREEN    = "\033[92m"
    WARNING    = "\033[93m"
    FAIL       = "\033[91m"
    BOLD       = "\033[1m"
    UNDERLINE  = "\033[4m"
    ENDC       = "\033[0m"
    
    def format_message(message, color):
        return f'{color}{message}{bcolors.ENDC}'
    
    def fail(message):
        return bcolors.format_message(message, bcolors.FAIL)
    
    
class logview: 
    
    def print_default(irandom: int=0, value:str=None):
        """
        Print default descriptions
        Args:
            irandom (int, 4 digits) random number to identify the code location
            value (str) description
        """ 
        s       = str(irandom).ljust(4, '.')     
        print(datetime.now().strftime('%H:%M:%S') + " " + s + " " + str(value))
        
    def print_key_value(irandom: int=0, key:str=None, value:str=None):
        """
        Print Key:Value descriptions
        Args:
            irandom (int, 4 digits): random number to identify the code location
            key (str, optional): 
            value (str, optional): description
        """
        ausgabe = str(key).ljust(50, '.')
        s       = str(irandom).ljust(4, '.') 
        print(datetime.now().strftime('%H:%M:%S') + " " + s + "    " + ausgabe + ": " + str(value))

    def print_error_value( irandom: int=0, value: str=None):
        """
        Print default descriptions
        Args:
            irandom (int, 4 digits) random number to identify the code location
            value (str) description
        """
        s       = str(irandom).ljust(4, '.')
        ausgabe = datetime.now().strftime('%H:%M:%S') + " " + s + " ERROR: " + str(value)
        print(logview._format_message(ausgabe, bcolors.FAIL))
        
    def _format_message(message, color):
        return f'{color}{message}{bcolors.HEADER}'