import re


text =  """
        .param 1wewe=2323 r232=123
        + dsd=123e-4
        +dsdsd=123e-4
        +dsd = 123e+10
        +dsd= 12310
        """

rx = re.findall(r'(?P<key>\w+)\s*=\s*(?P<value>[^ |\n]*)',text)

print(rx)