#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:10:25 2017

@author: manniarora
"""
import csv
import logging
import os
from datetime import date
import string
import re
from pdfminer.converter import TextConverter
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from io import StringIO
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#configurations
import csv
#----------------------------------------------------------------------
def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data)
#----------------------------------------------------------------------

def get_env_var(var, default):
    try:
        env_var = os.environ[var]
        return env_var
    except:
        return default


def isfile(path):
    return os.path.isfile(path)

# Regular expressinos used
bullet = r"\(cid:\d{0,2}\)"
email = r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}"
def get_phone(i,j,n):
  
  return r"\(?(\+)?(\d{1,3})?\)?[\s-]{0,1}?(\d{"+str(i)+"})[\s\.-]{0,1}(\d{"+str(j)+"})[\s\.-]{0,1}(\d{"+str(n-i-j)+"})"

not_alpha_numeric = r'[^a-zA-Z\d]'
number = r'\d+'

pincodeEx = r"[^\d]"+not_alpha_numeric+"(\d{6})"+not_alpha_numeric

months_short = r'(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)'
months_long = r'(january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december)'
month = r'('+months_short+r'|'+months_long+r')'
year = r'((20|19)(\d{2})|(\d{2}))'
start_date = month+not_alpha_numeric+r"?"+year
end_date = r'(('+month+not_alpha_numeric+r"?"+year+r')|(present))'+not_alpha_numeric
longer_year = r"((20|19)(\d{2}))"
year_range = longer_year+not_alpha_numeric+r"{1,3}"+longer_year
date_rangeEx =  r"("+start_date+not_alpha_numeric+r"{1,3}"+end_date+r")|("+year_range+r")"

def pdf_to_txt(file_name):
  try:
    file_pointer = open(file_name,'rb')
    parser = PDFParser(file_pointer)
    doc = PDFDocument()
    parser.set_document(doc)
    doc.set_parser(parser)
    doc.initialize('')
    # Setting up pdf reader
    pdf_resource_manager = PDFResourceManager()
    return_string = StringIO()
    #codec = 'utf-8'
    laparams = LAParams()
    
    device = TextConverter(pdf_resource_manager, return_string,laparams=laparams)
    interpreter = PDFPageInterpreter(pdf_resource_manager, device)
    
    for page in doc.get_pages():
      interpreter.process_page(page)
    file_pointer.close()
    device.close()
    # Get full string from PDF
    pdf_txt = return_string.getvalue()
    
    return_string.close()

    # logging.debug(pdf_txt)

    # Formatting removing and replacing special characters
    
    pdf_txt = pdf_txt.replace("\r", "\n")
    pdf_txt = re.sub(bullet,' ', pdf_txt)
    
    
    return pdf_txt

  except Exception as e:
    logging.error('Error converting pdf to txt: '+str(e))
    return ''
def fetch_email(resume_text):
  try:
    regular_expression = re.compile(email, re.IGNORECASE)
    emails = []
    result = re.search(regular_expression, resume_text)
    while result:
      emails.append(result.group())
      resume_text = resume_text[result.end():]
      result = re.search(regular_expression, resume_text)
    return emails
  except Exception as e:
    logging.error('Issue parsing email: ' + str(""))
    return []
def fetch_phone(resume_text):
  try:
    regular_expression = re.compile(get_phone(3, 3, 10), re.IGNORECASE)
    result = re.search(regular_expression, resume_text)
    phone = ''
    if result:
      result = result.group()
      for part in result:
        if part:
          phone += part
    if phone is '':
      for i in range(1, 10):
        for j in range(1, 10-i):
          regular_expression =re.compile(get_phone(i, j, 10), re.IGNORECASE)
          result = re.search(regular_expression, resume_text)
          if result:
            result = result.groups()
            for part in result:
              if part:
                phone += part
          if phone is not '':
            return phone
    return phone
  except Exception as e:
    logging.error('Issue parsing phone number: ' + resume_text +
      str(""))

def calculate_experience(resume_text):
  #
  def get_month_index(month):
    month_dict = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    return month_dict[month.lower()]

  try:
    experience = 0
    start_month = -1
    start_year = -1
    end_month = -1
    end_year = -1
    regular_expression = re.compile(date_rangeEx, re.IGNORECASE)
    regex_result = re.search(regular_expression, resume_text)
    while regex_result:
      date_range = regex_result.group()
      year_regex = re.compile(year)
      year_result = re.search(year_regex, date_range)
      if (start_year == -1) or (int(year_result.group()) <= start_year):
        start_year = int(year_result.group())
        month_regex = re.compile(months_short, re.IGNORECASE)
        month_result = re.search(month_regex, date_range)
        if month_result:
          current_month = get_month_index(month_result.group())
          if (start_month == -1) or (current_month < start_month):
            start_month = current_month
      if date_range.lower().find('present') != -1:
        end_month = date.today().month # current month
        end_year = date.today().year # current year
      else:
        year_result = re.search(year_regex, date_range[year_result.end():])
        if (end_year == -1) or (int(year_result.group()) >= end_year):
          end_year = int(year_result.group())
          month_regex = re.compile(months_short, re.IGNORECASE)
          month_result = re.search(month_regex, date_range)
          if month_result:
            current_month = get_month_index(month_result.group())
            if (end_month == -1) or (current_month > end_month):
              end_month = current_month
      resume_text = resume_text[regex_result.end():]
      regex_result = re.search(regular_expression, resume_text)

    return end_year - start_year  # Use the obtained month attribute
  except Exception as e:
    logging.error('Issue calculating experience: '+str(""))
    return None


"""

Utility function that fetches Job Position from the resume.
Params: cleaned_resume Type: string
returns: job_positions Type:List

"""
def fetch_jobs(cleaned_resume):
  positions_path = 'job_positions/positions'
  with open(positions_path, 'rb') as fp:
    jobs = pickle.load(fp)

  job_positions = []
  positions = []
  for job in jobs.keys():
    job_regex = r'[^a-zA-Z]'+job+r'[^a-zA-Z]'
    regular_expression = re.compile(job_regex, re.IGNORECASE)
    regex_result = re.search(regular_expression, cleaned_resume)
    if regex_result:
      positions.append(regex_result.start())
      job_positions.append(job.capitalize())
  job_positions = [job for (pos, job) in sorted(zip(positions, job_positions))]

  # For finding the most frequent job category
  hash_jobs = {}
  for job in job_positions:
    if jobs[job.lower()] in hash_jobs.keys():
      hash_jobs[jobs[job.lower()]] += 1
    else:
      hash_jobs[jobs[job.lower()]] = 1

  # To avoid the "Other" category and 'Student' category from
  # becoming the most frequent one.
  if 'Student' in hash_jobs.keys():
    hash_jobs['Student'] = 0
  hash_jobs['Other'] = -1

  return (job_positions, max(hash_jobs, key=hash_jobs.get).capitalize())


"""

Utility function that fetches the skills from resume
Params: cleaned_resume Type: string
returns: skill_set Type: List

"""
def fetch_skills(cleaned_resume):
  with open('skills/skills', 'rb') as fp:
    skills = pickle.load(fp)

  skill_set = []
  for skill in skills:
    skill = ' '+skill+' '
    if skill.lower() in cleaned_resume:
      skill_set.append(skill)
  return skill_set


"""

Utility function that fetches degree and degree-info from the resume.
Params: resume_text Type: string
returns:
degree Type: List of strings
info Type: List of strings

"""
def fetch_qualifications(resume_text):
  degree_path = 'qualifications/degree'
  with open(degree_path, 'rb') as fp:
    qualifications = pickle.load(fp)

  degree = []
  info = []
  for qualification in qualifications:
    qual_regex = r'[^a-zA-Z]'+qualification+r'[^a-zA-Z]'
    regular_expression = re.compile(qual_regex, re.IGNORECASE)
    regex_result = re.search(regular_expression, resume_text)
    while regex_result:
      degree.append(qualification)
      resume_text = resume_text[regex_result.end():]
      lines = [line.rstrip().lstrip()
      for line in resume_text.split('\n') if line.rstrip().lstrip()]
      if lines:
        info.append(lines[0])
      regex_result = re.search(regular_expression, resume_text)
  return degree, info

# Constants
LINES_FRONT = 3
LINES_BACK = 3

# Methods
def get_avoid_organizations():
  with open('organizations/avoid_organizations') as fp:
    avoid_organizations = pickle.load(fp)
  return avoid_organizations

def get_organizations():
  with open('organizations/explicit_organizations') as fp:
    organizations = pickle.load(fp)
  return organizations
"""

Utility function that fetches extra information from the resume.
Params: resume_text Type: string
returns: extra_information Type: List of strings

"""
def fetch_extra(resume_text):
  with open('extra/extra', 'rb') as fp:
    extra = pickle.load(fp)

  extra_information = []
  for info in extra:
    extra_regex = r'[^a-zA-Z]'+info+r'[^a-zA-Z]'
    regular_expression = re.compile(extra_regex, re.IGNORECASE)
    regex_result = re.search(regular_expression, resume_text)
    while regex_result:
      extra_information.append(info)
      resume_text = resume_text[regex_result.end():]
      regex_result = re.search(regular_expression, resume_text)
  return extra_information
def clean_resume(resume_text):

  cleaned_resume = []

  # replacing newlines and punctuations with space
  resume_text =resume_text.replace('\t', ' ').replace('\n', ' ')
  for punctuation in string.punctuation:
    resume_text = resume_text.replace(punctuation, ' ')
  resume_text = resume_text.split()

  # removing stop words and Stemming the remaining words in the resume
  stemmer = SnowballStemmer("english")
  for word in resume_text:
    if word not in stopwords.words('english') and not word.isdigit():
      stemmer.stem(word)
      cleaned_resume.append(word.lower())#stemmer.stem(word))

  cleaned_resume = ' '.join(cleaned_resume)
  return cleaned_resume


"""

Util function for fetch_employers module to get all the
organization names from the resume
Params: resume_text Type:String
Output: Set of all organizations Type: Set of strings

"""

"""

Util function for fetch_employers module to get employers
All organizations found near any job position is regarded as an employer
Params: resume_text Type:String
        job_positions Type: List of Strings
        organizations Type: List of Strings
        priority Type: Boolean True/False
Output: current_employers Type: List of strings
        all_employers Type: List of strings

"""
def fetch_employers_util(resume_text, job_positions, organizations):
  current_employers = []
  employers = []
  for job in job_positions:
    job_regex = r'[^a-zA-Z]'+job+r'[^a-zA-Z]'
    regular_expression = re.compile(job_regex, re.IGNORECASE)
    temp_resume = resume_text
    regex_result = re.search(regular_expression, temp_resume)
    while regex_result:

      # start to end point to a line before and after the job positions line
      # along with the job line
      start = regex_result.start()
      end = regex_result.end()
      lines_front = LINES_FRONT
      lines_back = LINES_BACK
      while lines_front != 0 and start != 0:
        if temp_resume[start] == '.':
          lines_front -= 1
        start -= 1
      while lines_back != 0 and end < len(temp_resume):
        if temp_resume[end] == '.':
          lines_back -= 1
        end += 1

      # Read from temp_resume with start and end as positions
      line = temp_resume[start:end].lower()

      for org in organizations:
        if org.lower() in line and org.lower() not in job_positions:
          if 'present' in line:
            if org.capitalize() in employers:
              employers.remove(org.capitalize())
            if org.capitalize() not in current_employers:
              current_employers.append(org.capitalize())
          elif org.capitalize() not in employers:
            employers.append(org.capitalize())

      temp_resume = temp_resume[end:]
      regex_result = re.search(regular_expression, temp_resume)

  return (current_employers, employers)


"""

Utility function that fetches the employers from resume
Params: resume_text Type: String
        job_positions Type: List of Strings
returns: employers Type: List of string

"""
def fetch_employers(resume_text, job_positions):

  # Cleaning up the text.
  # 1. Initially convert all punctuations to '\n'
  # 2. Split the resume using '\n' and add non-empty lines to temp_resume
  # 3. join the temp_resume using dot-space

  for punctuation in string.punctuation:
    resume_text = resume_text.replace(punctuation, '\n')

  temp_resume = []
  for x in resume_text.split('\n'):
    # append only if there is text
    if x.rstrip():
      temp_resume.append(x)

  # joined with dot-space
  resume_text = '. '.join(temp_resume)

  current_employers = []
  employers = []

  cur_emps, emps = fetch_employers_util(resume_text, job_positions,get_organizations())

  current_employers.extend(cur_emps)
  employers.extend(emps)

  cur_emps, emps = fetch_employers_util(resume_text, job_positions,
    fetch_all_organizations(resume_text))

  current_employers.extend([emp for emp in cur_emps
    if emp not in current_employers])
  employers.extend([emp for emp in emps
    if emp not in employers])

  return current_employers, employers
def fetch_address(resume_text):
  pincode_input_path = 'address/pincodes'
  address_input_path = 'address/pincode-district-state'
  states_input = 'address/states'
  district_state_input = 'address/district-states'
  pincodes = set()
  states = set()
  district_states = {}
  address = {}
  result_address = {}
  initial_resume_text = resume_text

  with open(pincode_input_path, 'rb') as fp:
    pincodes = pickle.load(fp)
  with open(address_input_path, 'rb') as fp:
    address = pickle.load(fp)

  regular_expression = re.compile(pincodeEx)
  regex_result = re.search(regular_expression, resume_text)
  while regex_result:
    useful_resume_text = resume_text[:regex_result.start()].lower()
    pincode_tuple = regex_result.group()
    pincode = ''
    for i in pincode_tuple:
      if (i <= '9') and (i >= '0'):
        pincode += str(i)
    if pincode in pincodes:
      result_address['pincode'] = pincode
      result_address['state'] = address[pincode]['state'].title()
      result_address['district'] = address[pincode]['district'].title()
      return result_address

    result_address.clear()
    resume_text = resume_text[regex_result.end():]
    regex_result = re.search(regular_expression, resume_text)

  resume_text = initial_resume_text.lower()

  with open(states_input, 'rb') as fp:
    states = pickle.load(fp)
  with open(district_state_input, 'rb') as fp:
    district_states = pickle.load(fp)

  # Check if the input is a separate word in resume_text
  def if_separate_word(pos, word):
    if (pos != 0) and resume_text[pos-1].isalpha():
      return False
    final_pos = pos+len(word)
    if ( final_pos !=len(resume_text)) and resume_text[final_pos].isalpha():
      return False
    return True

  result_state = ''
  state_pos = len(resume_text)
  result_district = ''
  district_pos = len(resume_text)
  for state in states:
    pos = resume_text.find(state)
    if (pos != -1) and(pos < state_pos) and if_separate_word(pos, state):
      state_pos = pos
      result_state = state
  for district in district_states.keys():
    pos = resume_text.find(district)
    if (pos != -1) and (pos < district_pos) and if_separate_word(pos, district):
      district_pos = pos
      result_district = district
  if (result_state is '') and (result_district is not ''):
    result_state = district_states[result_district]

  result_address['pincode'] = ''
  result_address['district'] = result_district.title()
  result_address['state'] = result_state.title()
  return result_address

"""

Utility function that fetches the Person Name from resume
Params: resume_text Type: string
returns: name Type: string

Returns the first noun (tried Person entity but couldn't make it work)
found by tokenizing each sentence
If no such entities are found, returns "Applicant name couldn't be processed"

"""
def fetch_name(resume_text):
  tokenized_sentences = nltk.sent_tokenize(resume_text)
  for sentence in tokenized_sentences:
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence), tagset='universal')):
      if hasattr(chunk, 'label'):# and chunk.label() == 'PERSON':
        chunk = chunk[0]
      (name, tag) = chunk
      if tag == 'NOUN':
        return name

  return "Applicant name couldn't be processed"
def fetch_pdf_urls(file_name):
  try:
    links = []
    file_pointer = open(file_name,'rb')
    parser = PDFParser(file_pointer)
    doc = PDFDocument()
    parser.set_document(doc)
    doc.set_parser(parser)
    doc.initialize('')
    
    # fetches URLs
    for page in doc:
      if 'Annots' in page.attrs.keys():
        link_object_list = page.attrs['Annots']
        # Due to implementation of pdfminer the link_object_list can either
        # be the list directly or a PDF Object reference
        if type(link_object_list) is not list:
          link_object_list = link_object_list.resolve()
        for link_object in link_object_list:
          if type(link_object) is not dict:
            link_object = link_object.resolve()
          if link_object['A']['URI']:
            links.append(link_object['A']['URI'])
    file_pointer.close()
    return links

  except Exception as e:
    logging.error('Error while fetching URLs : '+str(e))
    return ''
def fetch_all_organizations(resume_text):
  organizations = set()
  tokenized_sentences = nltk.sent_tokenize(resume_text)

  # Custom grammar with NLTK
  # NP - Noun Phrase
  # NN - Noun
  # NNP - Proper Noun
  # V - Verb
  # JJ - Adjective

  # In a sentence that contains NN NNNP V NN NN JJ NN.
  # The noun-phrases fetched are:
  # NP: NN NNP
  # NP: NN NN
  # NP: NN

  # Ex, "Application Developer at Delta Force"
  # => ["Application Developer", "Delta Force"]

  grammar = r"""NP: {<NN|NNP>+}"""
  parser = nltk.RegexpParser(grammar)

  avoid_organizations = get_avoid_organizations()

  for sentence in tokenized_sentences:

    # tags all parts of speech in the tokenized sentences
    tagged_words = nltk.pos_tag(nltk.word_tokenize(sentence))

    # then chunks with customize grammar
    # np_chunks are instances of class nltk.tree.Tree
    np_chunks = parser.parse(tagged_words)
    noun_phrases = []

    for np_chunk in np_chunks:
      if isinstance(np_chunk, nltk.tree.Tree) and np_chunk.label() == 'NP':
        # if np_chunk is of grammer 'NP' then create a space seperated string of all leaves under the 'NP' tree
        noun_phrase = ""
        for (org, tag) in np_chunk.leaves():
          noun_phrase += org + ' '

        noun_phrases.append(noun_phrase.rstrip())

    # Using name entity chunker to get all the organizations
    chunks = nltk.ne_chunk(tagged_words)
    for chunk in chunks:
      if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'ORGANIZATION':
        (organization, tag) = chunk[0]

        # if organization is in the noun_phrase, it means that there is a high chance of noun_phrase containing the employer name
        # eg, Delta Force is added to organizations even if only Delta is recognized as an organization but Delta Force is a noun-phrase
        for noun_phrase in noun_phrases:
          if organization in noun_phrase and organization not in avoid_organizations:
            organizations.add(noun_phrase.capitalize())

  return organizations


raw_text = pdf_to_txt('a.pdf')
name = fetch_name(raw_text)
emails = fetch_email(raw_text)
phone_numbers = fetch_phone(raw_text)
address = fetch_address(raw_text)
experience = calculate_experience(raw_text)
cleaned_resume = clean_resume(raw_text)
skills = fetch_skills(cleaned_resume)
(qualifications,degree_info) = fetch_qualifications(raw_text)
job_positions, category = fetch_jobs(cleaned_resume)
#current_employers,employers = fetch_employers(raw_text,job_positions)
extra_info = fetch_extra(raw_text)
#current_employers,employers = fetch_employers(raw_text,job_positions)
data = [name,str(emails[0]),phone_numbers,address["pincode"],address["district"],address["state"],experience,skills,job_positions,str(qualifications[0]),degree_info,]
path = "output.csv"
csv_writer(data, path)

