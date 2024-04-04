import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

features=[
    "Are you interested in learning how machines work and how to design them?",
"Do you enjoy working with computers and exploring new software applications?",
"Are you curious about how technology can be used to solve real-world problems?"
,"Would you like to explore careers in fields like robotics, aerospace, or software development?"
,"Are you comfortable with mathematical concepts and analytical thinking required for engineering?"
,"Do you enjoy building things and experimenting with different materials and designs?"
,"Are you interested in learning coding languages to create websites or software programs?"
,"Would you like to work in a field where you can contribute to advancements in technology and innovation?"
,"Are you willing to pursue further education or certifications to specialize in a specific area of engineering?"
,"Do you see yourself enjoying a career where you can apply scientific principles to practical solutions?"
,"Are you passionate about helping people and improving their health and well-being?"
,"Are you comfortable with the idea of working in environments like hospitals or clinics?"
,"Do you have a strong stomach and are not easily squeamish around blood or bodily fluids?"
,"Are you interested in learning about human anatomy, physiology, and diseases?"
,"Would you like to pursue a career that involves continuous learning and keeping up with advancements in medical science?"
,"Do you have good communication skills and enjoy interacting with patients and their families?"
,"Are you willing to dedicate several years to education and training to become a healthcare professional?"
,"Are you interested in exploring different healthcare fields, such as nursing, physical therapy, or medical research?"
,"Do you have the resilience and emotional intelligence to handle the challenges of working in healthcare?"
,"Do you see yourself making a positive impact on people's lives through a career in medicine or healthcare?"
,"Do you enjoy expressing yourself creatively through art, writing, or performance?"
,"Are you interested in learning about different cultures and historical periods?"
,"Would you like to pursue a career in fields like literature, fine arts, or theater?"
,"Do you have a passion for storytelling and communicating ideas through various mediums?"
,"Are you open to exploring unconventional career paths in creative industries?"
,"Do you see yourself thriving in environments where you can collaborate with other artists and professionals?"
,"Are you willing to take risks and embrace failure as part of the creative process?"
,"Do you enjoy analyzing and critiquing works of art, literature, or film?"
,"Are you interested in exploring careers that require strong communication and presentation skills?"
,"Do you see yourself making a living doing something you're passionate about in the arts and humanities field?"
,"Are you interested in understanding how laws are created and enforced in society?"
,"Do you enjoy debating and analyzing complex issues from different perspectives?"
,"Are you interested in learning about the principles of justice and ethics? "
,"Would you like to pursue a career as a lawyer, judge, or legal consultant? "
,"Do you have strong critical thinking and research skills required for legal analysis? "
,"Are you comfortable with public speaking and presenting arguments in front of others? "
,"Are you willing to dedicate several years to education and training to become a legal professional? "
,"Are you interested in exploring various areas of law, such as criminal law, civil law, or international law? "
,"Do you see yourself advocating for the rights of individuals and marginalized communities through legal practice? "
,"Do you have the integrity and commitment to uphold the principles of justice and fairness in your work? "
,"Are you passionate about playing sports and staying physically active?"
,"Do you enjoy working as part of a team towards a common goal? "
,"Are you interested in learning about sports science and training techniques to improve performance? "
,"Would you like to pursue a career as a professional athlete, coach, or sports therapist? "
,"Do you have the discipline and dedication to maintain a rigorous training regimen? "
,"Are you comfortable with the competitive nature of sports and the possibility of facing setbacks and injuries? "
,"Are you interested in exploring different sports and finding one that aligns with your strengths and interests? "
,"Do you see yourself inspiring others through your achievements and dedication to sports? "
,"Are you willing to pursue opportunities for scholarships or sponsorships to support your athletic pursuits? "
,"Do you have the resilience and determination to overcome challenges and pursue excellence in sports? "
,"Are you passionate about providing excellent customer service and creating memorable experiences for others? "
,"Do you enjoy working in fast-paced environments and interacting with people from diverse backgrounds? "
,"Are you interested in learning about different cultures, cuisines, and travel destinations? "
,"Would you like to pursue a career in hospitality management, hotel administration, or tourism marketing? "
,"Do you have strong interpersonal and communication skills required for working in the hospitality industry? "
,"Are you comfortable with multitasking and problem-solving in dynamic situations? "
,"Are you interested in exploring opportunities for internships or part-time jobs in hotels, restaurants, or travel agencies? "
,"Do you see yourself thriving in roles that involve coordinating events, managing accommodations, or organizing travel itineraries? "
,"Are you willing to adapt to changing trends and technologies in the hospitality and tourism sectors? "
,"Do you have a passion for creating positive experiences and ensuring customer satisfaction in the hospitality industry?"
,"Are you passionate about working with children or young adults? "
,"Do you enjoy sharing knowledge and helping others learn? "
,"Are you patient and empathetic towards diverse learners' needs? "
,"Would you like to pursue a career as a teacher, instructor, or education administrator? "
,"Do you have strong communication and interpersonal skills for engaging with students and parents? "
,"Are you interested in learning about educational theories and teaching methods? "
,"Are you willing to pursue further education and training, such as a teaching certification or graduate degree in education? "
,"Do you see yourself making a positive impact on students' lives through teaching and mentoring? "
,"Are you open to adapting teaching strategies to accommodate different learning styles and abilities? "
,"Do you have a passion for fostering a supportive and inclusive learning environment in the classroom? "
,"Are you interested in learning about how businesses operate and succeed in competitive markets? "
,"Do you enjoy taking initiative and pursuing opportunities for innovation and growth? "
,"Are you comfortable with taking calculated risks and managing uncertainties in business ventures? "
,"Would you like to pursue a career as an entrepreneur, business owner, or corporate leader? "
,"Do you have strong analytical and problem-solving skills for identifying market trends and strategic opportunities? "
,"Are you interested in learning about different aspects of business management, such as marketing, finance, or operations? "
,"Are you willing to invest time and resources into developing your business ideas and concepts? "
,"Do you see yourself building networks and partnerships to support your entrepreneurial endeavors? "
,"Are you open to learning from failures and setbacks to iterate and improve your business strategies? "
,"Do you have the ambition and determination to pursue your entrepreneurial dreams despite challenges and obstacles? "
,"Are you fascinated by human behavior and the workings of the mind? "
,"Do you enjoy helping others navigate through challenges and improve their mental well-being? "
,"Are you empathetic and nonjudgmental towards people's experiences and emotions? "
,"Would you like to pursue a career as a psychologist, counselor, or therapist? "
,"Do you have strong listening and communication skills for building trust and rapport with clients? "
,"Are you interested in learning about various therapeutic approaches and interventions for supporting individuals' mental health? "
,"Are you willing to pursue advanced education and licensure requirements for practicing psychology or counseling? "
,"Do you see yourself working in diverse settings such as schools, clinics, or private practices to serve different populations? "
,"Are you committed to ongoing self-reflection and professional development to enhance your counseling skills? "
,"Do you have the resilience and self-care practices to manage the emotional demands of working in mental health professions? "
,"Are you interested in pursuing a career that involves working with financial data and analyzing business transactions?"
,"Do you enjoy exploring concepts related to finance, economics, and market dynamics?"
,"Are you comfortable with handling complex mathematical calculations and interpreting financial statements?"
,"Would you like to work in roles such as financial analysis, auditing, or investment banking?"
,"Are you passionate about understanding tax laws, regulations, and financial reporting standards?"
,"Do you have a strong attention to detail and the ability to identify discrepancies in financial documents?"
,"Are you willing to commit to continuous learning and staying updated on changes in the financial industry?"
,"Do you see yourself pursuing internships or work experiences in accounting firms or financial institutions?"
,"Are you interested in exploring opportunities to specialize and enhance your expertise in accounting and finance?"
,"Do you have the dedication and perseverance to overcome the challenges of pursuing a career in commerce or chartered accountancy?"
]

question_dict={}

df = pd.DataFrame()

def making_df():    
    for que in features:
        question_dict[que]=[None]
    global df
    df=pd.DataFrame(question_dict)

def addValues(que_lst):
    i=0
    record={}
    for sub in que_lst:
        its=sub.lstrip()
        its=sub.rstrip()
        record[sub]=int(1)
    df.iloc[0]=record

def preprocessor_2():   
   df_filled = df.fillna(0)
   columns_to_convert = df_filled.columns 
   df_filled[columns_to_convert] = df_filled[columns_to_convert].astype('int64')
   inp1=df_filled.loc[0]
#    inp_lst=[inp1[:len(inp1)-2]]
#    print(inp1)
   print(inp1.shape) 
   g = np.reshape([inp1],(1,100))
   print(g.shape)
   print(type(g))
   return g

def prediction(inp_lst):
   print("hello 1")
   loaded_model = load_model('my_model.h5')
   print("hello 2")
   prediction_result=loaded_model.predict(inp_lst)
   print("hello 3")
   return prediction_result





app = Flask(__name__)

@app.route('/predict_career', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the POST request
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Invalid input. Expected a JSON list of strings.'}), 400
    
    making_df()
    addValues(data)
    arr=prediction(preprocessor_2())
    max_index = max(range(len(arr[0])), key=arr[0].__getitem__)
    print(arr[0])
    print(max_index)
    Categories= ['Arts & Humanities', 'Business& Enterpreneurship', 'Commerce & CA', 'Education & Teaching', 'Engineering', 'Hospitality', 'Law and Legal Studies' , 'Medical & Heathcare','Psychology and Counseling', 'Sports & Athletics']
    print(Categories[max_index])
    # Process the list of strings (In this example, just concatenating them)
    result = ' '.join(data)
    
    # Return the result in JSON format
    ans=Categories[max_index]
    return jsonify({'result': ans})

if __name__ == '__main__':
    app.run(debug=True)