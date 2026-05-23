from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = "Shivam" 
    age: int = 22
    email: Optional[EmailStr] = None
    cgpa: float = Field(default=5.5, gt=0.0, lt=10.0, description="CGPA must be between 0.0 and 10.0")

# student1 = {'name': 'Bob', 'age': 22}
student1 = {"email":"shiva@domain.com", "cgpa": 8.5}

student = Student(**student1)
#Output Option 1. Model
print(student) 
print("=="*20)

#Output Option 2. Dict
student_dict = dict(student)
print(student_dict)
print(student_dict['name'])
print("=="*20)

# Output Option 3. JSON
student_json = student.model_dump_json()
print(student_json)
print("=="*20)
print(type(student.name))
print(type(student.age))
print(type(student.email))