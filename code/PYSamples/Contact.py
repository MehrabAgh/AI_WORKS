import re
all_contact = dict()

def create_contact(name , number , email):    
    return all_contact.update({"name":name , "number" : number , "email":email})

def CREATE_PAGE():
    print(chr(27) + "[2J")
    _name = input("Enter your name : ")
    _pn = input("Enter your Phone Number: ")
    _email = input("Enter your Email: ")
    create_contact( _name, _pn, _email)            
    f = open("allcontact.txt","a")
    f.write(str(all_contact)+',')
    f.close()                 
    print("user created")

def FIND_PAGE():
    print(chr(27) + "[2J")
    _nameTemp = str(input("Enter your name for find number : "))
    f = open("allcontact.txt","r")
    data = f.readline()         
    f.close()
    data = eval(data)    
    NEW_DATA = list(data)                                   
    for i in NEW_DATA:
        if (i['name'] == _nameTemp):
            print("\n ! Found User ! \n")
            print("Name :" , i['name'] , '\n')
            print("PhoneNumber :" , i['number'], '\n')
            print("Email:" , i['email'] , '\n')
            break
        elif (re.search(re.escape('all'),_nameTemp, re.IGNORECASE)):
            print( 'Name  : ' , i['name'] )
            print( 'PhoneNumber  : ' , i['number'] )
            print( 'Email  : ' , i['email'] )
            print('\n')                  
        else:
          print("\n ! User Not Found !")  

def UPDATE_PAGE():    
    print(chr(27) + "[2J")
    myUser = input("please enter name user for get information: ")

    f = open("allcontact.txt","r")
    data = f.readline()      
    f.close()     
    data = eval(data)    
    NEW_DATA = list(data)
    for i in NEW_DATA:
        if (i['name'] == myUser):
            myAttrb = input("please enter your attribute for update (name , number , email): ")
            myValue = input("please enter new value for attribute : ")
            i[str(myAttrb)] = myValue                    
            f = open("allcontact.txt","w")                    
            f.write(str(NEW_DATA)) 
            f.close() 
            break
        else:
            print("\n ! User Not Found !")            

    print("! your Information Updated !")

def DELETE_PAGE():
    print(chr(27) + "[2J")    
    myUser = input("please enter name user for get information: ")
    f = open("allcontact.txt","r")
    data = f.readline()      
    f.close()     
    print(data)
    data = eval(data)    
    NEW_DATA = list(data)
    for i in NEW_DATA:
        if (i['name'] == myUser): 
            state = input("Are you sure of your work (yes , no)?")
            if(state == 'yes'):
                NEW_DATA.remove(i)    
                f = open("allcontact.txt","w")                    
                f.write(str(NEW_DATA)) 
                f.close() 
                print("\n ! Your Information has been Deleted !")                                        
            else : break
            break
        else:
            print("\n ! User Not Found !")          
    

def Main_Application():
    while True:      
        print("\nWelcome to Contact_List App \n")    
        print("Enter Your Index Page:\n")    
        print("1 - Create New Contact \n")
        print("2 - Find a Contact \n")
        print("3 - Update a Contact \n")
        print("4 - Delete a Contact \n")
        print("0 - Exit \n")

        index = input("please enter page : ")

        match int(index):
            case 1:       
                CREATE_PAGE()     
            case 2:                          
                FIND_PAGE()
            case 3:        
                UPDATE_PAGE()       
            case 4:
                DELETE_PAGE()               
            case 0:
                print("# GOOD BYE USER #")
                break
            case _:    
                print("Page Not Found (404)")                    
        break



Main_Application()