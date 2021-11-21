
"""

TR:

Kütüphaneyi kullanmadan önce hspice programının path edildiğinden emin olunuz.

"""

import signal

import os , time , json , asyncio
import re
import subprocess
import itertools
import shlex 
from subprocess import Popen, PIPE

class Result(object):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
            
        return result
    return timed

class HSpicePy(object):
    """
    HSpicePy is a high-level wrapper for the HSpice simulator.
    It is designed to be used with the HSpicePy.py script.
    """

    def __init__(self, path :str,file_name : str,design_file_name : str,timeout :int , save_output : bool = True,loop=None):
        """
        Initializes the HSpicePy class.
        path -> path of the .sp file
        file_name -> ".sp" file name
        design_file_name -> ".cir" file name
        timeout -> timeout of the simulation
        """
        self.file_name = file_name if ".sp" not in file_name else file_name[:-3]
        self.design_file_name = design_file_name if ".cir" not in design_file_name else design_file_name[:-4]
        self.path = path
        self.timeout = timeout
        self.parameters_dict = {}
        self.parameters = None
        self.instructions = []
        self.save_output = save_output
        self.max_error = 50
        self.runner = False
        self.__mt0_result = None
        self.__result = None
        self.__operation_point_result = None
        self.loop = loop
    
    
    def set_parameters(self,**kwargs):
        """
        Sets the parameters of the HSpice simulation.
        """
        param = " "
        param = param.join(f"{key}={value}"for key,value in kwargs.items())
        param = ".PARAM " + param
        self.instructions.append(param)
        
    def get_parameters_from_cir(self): 
        """
        Gets the parameters from the cir file.

        
        TODO line ve alt satırlı
        """
        
        devamke = True
        max_error = self.max_error
        params = None
        while devamke:
            try:
                with open(f"{self.path}/{self.design_file_name}.cir","r") as file:
                    data = file.read()
                    params = re.findall(r'(?P<key>\w+)\s*=\s*(?P<value>[^ |\n]*)',data)
                    file.close()
                devamke = False
            except Exception as e:
                devamke = True
                max_error -= 1
                if max_error == 0:
                    return None

        parameters_dict = {}
        for key,value in params:
            parameters_dict[key] = value
        self.set_parameters(**parameters_dict)
    
    def change_parameters_to_cir(self,**kwargs):
        """
        Changes the parameters in the cir file.
        """
        # with open(f"{self.path}/{file_name}","r") as file:
        #     data = file.read()
        #     for key,value in self.parameters_dict.items():
        #         data = data.replace(key,value)
        
        params = ".PARAM\n"+"".join(f"+ {key} = {value}\n"for key,value in kwargs.items())
        with open(f"{self.path}\\{self.design_file_name}.cir","w+") as file:
            file.write(params)
            file.close()
    
    @property
    def result(self):
        """
        Returns the result of the simulation.
        """
        return self.__result
    
    @property
    def operation_point_result(self):
        """
        Returns the result of the simulation.
        """
        return self.__operation_point_result

    @property
    def mt0_result(self):
        """
        Returns the result of the simulation.
        """
        return self.__mt0_result

    # @result.setter
    # def result(self,value):
    #     self.__result = value
    
    # @result.deleter
    # def result(self):
    #     self.__result = None
    def __get_dp0_log(self):
        
        try:
            if not self.runner:
                return
            dp0_log = {}
            file_name = self.file_name + ".dp0"
            tables = None
            variables = []
            devamke = True
            max_error = self.max_error
            while devamke:
                try:
                    with open(f"{self.path}\\{file_name}","r") as dp0:
                        # data = dp0.read()
                        tables = dp0.readlines()
                        tables_ = {} 
                        i = 0
                        new_table = False
                        table_number = 0
                        for line in tables:
                            if len(line) < 5:continue
                            line = line.replace("\n","")
                            if line.startswith("-") and not new_table:
                                new_table = True
                                table_number += 1
                                table_line_number = 0
                                
                                continue
                            if line.startswith("|-") and "+" not in line:
                                new_table = False

                            if new_table:
                                if table_line_number == 0:
                                    params = list(filter(None, line.split("|")))
                                    
                                    tables_[params[0].strip()] = {key.strip() :{} for key in params[1:]}
                                    
                                if table_line_number > 1:
                                    variables = list(filter(None, line.split("|")))
                                    variable_name = variables[0].strip()
                                    variables = variables[1:]
                                    i = 0
                                    for key in params[1:]:
                                        tables_[params[0].strip()][key.strip()][variable_name] = variables[i].strip()
                                        i+=0
                                table_line_number += 1
                    devamke = False
                except Exception as e:
                    devamke = True
                    max_error -= 1
                    if max_error == 0:
                        return None

                    
            if self.save_output:
                with open(f"{self.path}\\out\\{file_name[:-3]}_dp0.json","w") as outfile:
                    json.dump(tables_, outfile)       
            self.__operation_point_result = tables_
        except Exception as e:
            print(e)           

    def __get_ma0_log(self):
        """
        TR
        -
        Simülasyon oluşturulduktan sonra oluşan .ma0 dosyasını okur ve çıktıları kaydeder.
        var = [bw,gain , hreal]
        res = [7,3,-9]
        """
        try: 
            if not self.runner:
                return    
            file_name = self.file_name + ".ma0"
            lines = None
            while not lines:
                with open(f"{self.path}\\{file_name}","r") as ma0:
                    # data = ma0.read()
                    
                    lines =  ma0.readlines()
                    ma0.close()
            variables = lines[-2]
            results = lines[-1]
            res = {variable:result for variable,result in zip(variables.split(),results.split())}
            if self.save_output:
                with open(f"{self.path}\\out\\{file_name[:-3]}_ma0.json","w") as outfile:
                    json.dump(res, outfile)   
            self.__result = res
        except Exception as e:
            print(e)
            self.__result = None
        
    def __get_mt0_log(self):
        """
        TR
        -
        Simülasyon oluşturulduktan sonra oluşan .mt0 dosyasını okur ve çıktıları kaydeder.
        var = [bw,gain , hreal]
        res = [7,3,-9]
        """
        try: 
            if not self.runner:
                return    
            file_name = self.file_name + ".mt0"
            lines = None
            maks = self.max_error
            i = 0
            while not lines:
                with open(f"{self.path}\\{file_name}","r") as mt0:
                    # data = ma0.read()
                    
                    lines =  mt0.readlines()
                    mt0.close()
                i+=1
                if i > maks:
                    break
            variables = lines[-2]
            results = lines[-1]
            res = {variable:result for variable,result in zip(variables.split(),results.split())}
            if self.save_output:
                with open(f"{self.path}\\out\\{file_name[:-3]}_mt0.json","w") as outfile:
                    json.dump(res, outfile)   
            self.__mt0_result = res
        except Exception as e:
            print(e)
            self.__mt0_result = None
    async def killer(self,x):
        await asyncio.sleep(2)
        x.kill()
    # @timeit
    def run(self):
        try:
            file_name = self.file_name if ".sp" in self.file_name else self.file_name+".sp"
            
            # subprocess.run([r'hspicerf', f"{self.path}\\{file_name}"],cwd=self.path+f"\\out")
            path = self.path.replace("\\","/")
            cwd = path+f"/out"
            cmd = shlex.split(f"start/min/wait hspicerf {path}/{file_name}")
            # p = subprocess.call(cmd,cwd=cwd,shell=True)
            process = None
            
            process=Popen(cmd,shell=True, stdout=PIPE, stderr=PIPE, cwd=self.path+f"\\out")
            stdout, stderr = process.communicate()
            time.sleep(1)
            process.kill()

            # os.system(f"hspicerf -ahv {self.path}//{file_name} {self.path}//output")   ,f" > "
            # print(stdout,  "___" ,stderr)
            self.__get_ma0_log()
            self.__get_dp0_log()
            self.__get_mt0_log()
        except Exception as e:
            print(e)
            pass
        
    async def run_async(self):
        try:
            file_name = self.file_name if ".sp" in self.file_name else self.file_name+".sp"
            # try:
            #     proc = None
            #     async def shell():
            #         nonlocal proc
            #         proc = await asyncio.create_subprocess_shell(f"hspicerf {self.path}\\{file_name}", cwd=self.path+f"\\out")
                
            #     await asyncio.wait_for(shell(),timeout=0.3)
            #     proc.kill()
            # # stdout, stderr = asyncio.gather( proc.communicate())
            
                
            # # await proc.kill()
            # # await proc.terminate()
            # except asyncio.TimeoutError:
            #     await proc.kill()
            proc = await asyncio.create_subprocess_shell(f"start/min/wait hspicerf {self.path}\\{file_name}", cwd=self.path+f"\\out")    
            self.__get_ma0_log()
            self.__get_dp0_log()
            self.__get_mt0_log()

            
        except Exception as e:
            print(e)
            pass    
        
        
#Set-ExecutionPolicy Unrestricted


meAbsPath = os.path.dirname(os.path.realpath(__file__))
h = HSpicePy(file_name="amp",design_file_name="designparam",path=meAbsPath,timeout="")



# print(h.result)
# loop = asyncio.get_event_loop()
asyncio.run(h.run_async())
print(h.result)
# h.set_parameters(R1=1,R2=2,R3=3)


# h.get_parameters_from_cir((meAbsPath +"\d.cir"))
# print(h.parameters_dict)

