import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
import psycopg2
import mysql.connector
from langchain_groq import ChatGroq
import urllib.parse

# إعداد صفحة Streamlit
st.set_page_config(page_title="التفاعل مع قاعدة البيانات باستخدام LangChain", page_icon="🦜")
st.title("🦜 التفاعل مع قاعدة البيانات باستخدام LangChain")

# تعريف الثوابت لأنواع قواعد البيانات
LOCALDB = "LOCAL_SQLITE"
POSTGRESQL = "POSTGRESQL_CONNECTION"
MYSQLDB = "MYSQL_CONNECTION"

# اختيار قاعدة البيانات من الشريط الجانبي
options = ["قاعدة بيانات SQLite - Student.db", "الاتصال بقاعدة بيانات PostgreSQL", "الاتصال بقاعدة بيانات MySQL"]
selected_db_option = st.sidebar.radio("اختر قاعدة البيانات التي تريد التفاعل معها", options)

# تعريف المتغيرات الخاصة بالاتصال
db_connection = None
pg_host, pg_user, pg_password, pg_db = None, None, None, None
mysql_host, mysql_user, mysql_password, mysql_db = None, None, None, None

# جمع معلومات الاتصال بناءً على اختيار المستخدم
if selected_db_option == options[0]:  # SQLite
    db_connection = LOCALDB
elif selected_db_option == options[1]:  # PostgreSQL
    db_connection = POSTGRESQL
    pg_host = st.sidebar.text_input("مضيف PostgreSQL", value="127.0.0.1").strip()
    pg_user = st.sidebar.text_input("مستخدم PostgreSQL", value="postgres").strip()
    pg_password = st.sidebar.text_input("كلمة مرور PostgreSQL", type="password")
    pg_db = st.sidebar.text_input("اسم قاعدة البيانات PostgreSQL").strip()
elif selected_db_option == options[2]:  # MySQL
    db_connection = MYSQLDB
    mysql_host = st.sidebar.text_input("مضيف MySQL").strip()
    mysql_user = st.sidebar.text_input("مستخدم MySQL").strip()
    mysql_password = st.sidebar.text_input("كلمة مرور MySQL", type="password")
    mysql_db = st.sidebar.text_input("اسم قاعدة البيانات MySQL").strip()

# جمع مفتاح API الخاص بـ Groq
api_key = st.sidebar.text_input("مفتاح API لـ Groq", type="password")

# التحقق من وجود معلومات الاتصال ومفتاح API
if not db_connection:
    st.info("يرجى إدخال معلومات الاتصال بقاعدة البيانات.")
if not api_key:
    st.info("يرجى إضافة مفتاح API لـ Groq.")

# إعداد النموذج الخاص بـ LLM باستخدام Groq
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# دالة لإعداد الاتصال بقاعدة البيانات
@st.cache_resource(ttl="2h")
def setup_database_connection(db_connection, pg_host=None, pg_user=None, pg_password=None, pg_db=None, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    """تقوم بإرجاع كائن SQLDatabase بناءً على إعدادات الاتصال المحددة."""
    try:
        if db_connection == LOCALDB:
            db_filepath = (Path(__file__).parent / "student.db").absolute()
            creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
            return SQLDatabase(create_engine("sqlite:///", creator=creator))

        elif db_connection == POSTGRESQL:
            if not (pg_host and pg_user and pg_password and pg_db):
                raise ValueError("❌ يرجى تقديم جميع تفاصيل الاتصال بـ PostgreSQL.")
            encoded_password = urllib.parse.quote(pg_password)
            connection_url = f"postgresql+psycopg2://{pg_user}:{encoded_password}@{pg_host}/{pg_db}"
            return SQLDatabase(create_engine(connection_url))

        elif db_connection == MYSQLDB:
            if not (mysql_host and mysql_user and mysql_password and mysql_db):
                raise ValueError("❌ يرجى تقديم جميع تفاصيل الاتصال بـ MySQL.")
            connection_url = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            return SQLDatabase(create_engine(connection_url))

    except Exception as e:
        st.error(f"❌ فشل الاتصال: {e}")
        st.stop()

# إعداد الاتصال بقاعدة البيانات
db = setup_database_connection(db_connection, pg_host, pg_user, pg_password, pg_db, mysql_host, mysql_user, mysql_password, mysql_db)

# إعداد Toolkit للتفاعل مع قاعدة البيانات
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# إنشاء وكيل SQL للتفاعل مع قاعدة البيانات
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# تاريخ المحادثة
if "messages" not in st.session_state or st.sidebar.button("مسح تاريخ المحادثة"):
    st.session_state["messages"] = [{"role": "assistant", "content": "كيف يمكنني مساعدتك؟"}]

# عرض تاريخ المحادثة
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# إدخال استعلام المستخدم
user_input = st.chat_input(placeholder="اطرح أي سؤال من قاعدة البيانات")

# التعامل مع استعلامات المستخدم
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        callback_handler = StreamlitCallbackHandler(st.container())
        response = agent.run(user_input, callbacks=[callback_handler])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
