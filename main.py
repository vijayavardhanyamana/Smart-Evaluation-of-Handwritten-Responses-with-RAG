from HTR.app import extract_text_from_image
from correct_answer_generation.answer_generation_database_creation import database_creation, answer_generation


from similarity_check.tf_idf.tf_idf_score import create_tfidf_values, tfidf_answer_score
from similarity_check.semantic_meaning_check.semantic import similarity_model_score, fasttext_similarity

def new_value(value, old_min, old_max, new_min, new_max):
    new_value = new_min + ((value - old_min) * (new_max - new_min)) / (old_max - old_min)
    return new_value


# handwritten answer image path
img_path = "ans_image/1.jpg"
answer = extract_text_from_image(img_path)
print(answer)

query = "What is an operating system? Explain its main functions."

# pdf1 path
path1 = "Knowledge_Retriever_pdf/OS1.pdf"
database_creation(path1)
correct_answer1 = answer_generation(path1,query)
print(correct_answer1)

# pdf2 path
path2 = "Knowledge_Retriever_pdf/OS2.pdf"
database_creation(path2)
correct_answer2 = answer_generation(path2,query)
print(correct_answer2)


# correct_answer1 = "An operating system is a collection of system programs that together control the operations of a computer system. Some examples of operating systems are UNIX, Mach, MS-DOS, MS-Windows, WindowsNT, Chicago, OS2, MacOS, VMS, MVS, and VM System View: From the computer's point of view, an operating system is a control program that manages the execution of user programs to prevent errors and improper use of the computer. It is concerned with the operation and control of IO devices."

# correct_answer2 = "The operating system sits between the user and the hardware of the computer providing an operational environment to the users and application programs. For a user, therefore, a computer is nothing but the operating system running on it. It is an extended machine Operating System (or briefly OS) provides services to both the users and to the programs 1. It provides programs, an environment to execute OS is a resource allocate Manages all resources Decides between conflicting requests for efficient and fair resource use OS is a control program Controls execution of programs to prevent errors and improper use of the computer."
# answer = "An operating system is system software that manages Computer hardware and software resources and provides common services for computer programs Examples include UNIX, Linux, windows and tacos It ensures process scheduling management and device Communication, enabling multitasking and secure execution used applications the is acts as a control program to prevent improper use of resources and optimize system performance also plays a crucial in coordinating input output operations ensuring seamless Communication between hardware and software components"
# answer = "An operating system is a collection of software that manages hardware and software resources in a computer. It provides a platform for applications to run and coordinates processes, memory, and storage. Examples include UNIX, Windows, and macOS, which enable multitasking and secure user operations. By controlling hardware like I/O devices and ensuring the efficient execution of programs, the OS acts as an intermediary between users and the machine. It plays a crucial role in error prevention and system reliability, allowing for smooth and efficient use of computing resources."
# answer = "An operating system is a program designed to load applications and manage the appearance of the desktop environment. It works primarily to enhance user experience by controlling graphical features like icons, themes, and window arrangements. For instance, operating systems such as iOS and Android mainly function as user interfaces for smartphones. They simplify accessing apps but do not handle complex operations like process management or security. The OS focuses on ensuring applications run visually appealingly, leaving performance optimization to hardware."
# answer = "An operating system is mainly used to open and close applications on a device. It helps users interact with the graphical interface and ensures that programs can load. While it is often confused with hardware, the OS’s primary role is to make computers visually appealing and user-friendly. For example, mobile operating systems like iOS focus on apps and user convenience but don’t manage lower-level tasks like memory or processor allocation. Without an OS, applications wouldn’t look the same or respond as quickly, but hardware would still operate independently."
# answer = "An operating system is a hardware component inside the computer that generates electricity to power other parts. It functions like a battery, storing energy and distributing it when needed. Examples of operating systems include the RAM, the power supply unit, and the motherboard, all of which are critical for maintaining a computer's energy flow. Without an OS, the computer would not turn on because it would lack the electrical charge necessary to power the system."
# answer = "An operating system is a physical object that helps connect computers to the internet by capturing Wi-Fi signals. It is stored inside the monitor and works by converting light signals into data that the computer can process. Examples of operating systems include routers, Bluetooth devices, and USB drives. Without an operating system, computers wouldn’t be able to browse websites or use social media. The OS also powers the screen by generating electricity and controls how bright or dim the display appears."
tf_idf_word_values, max_tfidf = create_tfidf_values(correct_answer1,correct_answer2)

marks = 0

marks1 = tfidf_answer_score(answer,tf_idf_word_values,max_tfidf,marks =10)

print(tfidf_answer_score,marks1)

if marks1>3:
    marks += new_value(marks1, old_min = 3, old_max=10, new_min=0, new_max=5)
    print("TFIDF Score",float(marks))
    
    
if marks1>2:
    marks2 = similarity_model_score(correct_answer1,correct_answer2,answer)
    a = 0
    if marks2>0.95:
        marks += 3
        a = a+3
    elif marks2>0.5:
        marks += new_value(marks2, old_min = 0.5, old_max=0.95, new_min=0, new_max=3)
        a = a+new_value(marks2, old_min = 0.5, old_max=0.95, new_min=0, new_max=3)
    print("sentence-transformers/all-MiniLM-L6-v2 with Cosine Similarity",a)

    marks3 = fasttext_similarity(correct_answer1,correct_answer2,answer)
    b = 0
    if marks2>0.9:
        marks += 2
        b= b+2
    elif marks3>0.4:
        marks += new_value(marks3, old_min = 0.4, old_max=0.9, new_min=0, new_max=2)
        b=b+new_value(marks3, old_min = 0.4, old_max=0.9, new_min=0, new_max=2)
    print("fasttext-wiki-news-subwords-300 with Soft Cosine Similarity",b)

print("final marks",float(marks))