import sys
def error_message_detail(error,error_detail:sys):
    _,_,exe_tb =error_detail.exc_info()
    error_message = f"Error occurred in python script name : [{exe_tb.tb_frame.f_code.co_filename}] at line number: [{exe_tb.tb_lineno}] with error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        super().__init__(error)
        self.error_message = error_message_detail(error,error_detail=error_detail)

    def __str__(self):
        return self.error_message
