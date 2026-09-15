import streamlit as st
import requests
import time
import pandas as pd
import os
import json
from datetime import datetime
import plotly.graph_objects as go
try:
    from pose_analyzer import PoseTracker
    HAS_POSE_TRACKER = True
except Exception:
    HAS_POSE_TRACKER = False

COZE_API_KEY = "pat_EvZ6I6I0luMEldZD7zuVAshog9OK5MSYknoRrCBDHmvA7hQdlWMN56cI3BoZjzQL"
COZE_SCRIPT_BOT_ID = "7642156410809384969"
COZE_PPT_BOT_ID = "7643667281079959561"

def save_to_excel(excel_file, student_name, user_script, script_report, ppt_filename, ppt_report, posture_metrics=None):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if posture_metrics:
        if posture_metrics.get("is_valid", True):
            g_score = posture_metrics.get("gesture_score", 0)
            s_score = posture_metrics.get("stability_score", 0)
            p_score = posture_metrics.get("posture_score", 0)
            i_score = posture_metrics.get("interaction_score", 0)
            o_score = posture_metrics.get("overall_score", 0)
        else:
            g_score = s_score = p_score = i_score = o_score = "未检出(质检未过)"
        p_report = posture_metrics.get("report", "无评语")
    else:
        g_score = s_score = p_score = i_score = o_score = "未评估"
        p_report = "未上传视频"

    new_record = {
        "诊断时间": [current_time],
        "学生姓名": [student_name],
        "说课稿字数": [len(user_script)],
        "说课稿内容": [user_script],
        "说课稿诊断报告": [script_report],
        "PPT文件名": [ppt_filename if ppt_filename else "未上传"],
        "PPT课件评估报告": [ppt_report if ppt_report else "未评估"],
        "教态综合分": [o_score],
        "手势活跃分": [g_score],
        "身体平稳分": [s_score],
        "站姿挺拔分": [p_score],
        "互动面向分": [i_score],
        "教态诊断报告": [p_report]
    }
    new_df = pd.DataFrame(new_record)

    if not excel_file.endswith(".xlsx"):
        excel_file += ".xlsx"

    if os.path.exists(excel_file):
        try:
            existing_df = pd.read_excel(excel_file)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_excel(excel_file, index=False)
        except Exception as e:
            st.error(f"写入Excel失败，请检查文件是否被占用：{str(e)}")
    else:
        new_df.to_excel(excel_file, index=False)

def upload_file_to_coze(api_key, uploaded_file):
    url = "https://api.coze.cn/v1/files/upload"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
    try:
        response = requests.post(url, headers=headers, files=files, timeout=60)
        res_data = response.json()
        if response.status_code == 200 and res_data.get("code") == 0:
            return res_data["data"]["id"], None
        return None, f"HTTP {response.status_code} | {response.text}"
    except Exception as e:
        return None, f"网络异常: {str(e)}"

def call_coze_agent(api_key, bot_id, user_message, file_id=None):
    url = "https://api.coze.cn/v3/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    if file_id:
        content_structure = [
            {"type": "text", "text": user_message},
            {"type": "file", "file_id": file_id}
        ]
        content_payload = json.dumps(content_structure)
        content_type = "object_string"
    else:
        content_payload = user_message
        content_type = "text"

    payload = {
        "bot_id": bot_id,
        "user_id": "streamlit_user_123",
        "stream": False,
        "additional_messages": [
            {"role": "user", "content": content_payload, "content_type": content_type}
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            return f"❌ HTTP 状态码异常 ({response.status_code}): {response.text}"
        
        res_data = response.json()
        if res_data.get("code") != 0:
            return f"❌ 扣子平台报错 (Code {res_data.get('code')}): {res_data.get('msg')}"

        chat_id = res_data["data"]["id"]
        conversation_id = res_data["data"]["conversation_id"]

        retrieve_url = f"https://api.coze.cn/v3/chat/retrieve?chat_id={chat_id}&conversation_id={conversation_id}"
        status = "in_progress"
        retry_count = 0
        while status in ["in_progress", "created"]:
            time.sleep(1.5)
            try:
                get_res = requests.get(retrieve_url, headers=headers, timeout=15)
                status = get_res.json().get("data", {}).get("status")
                retry_count = 0  # 只要有一次成功，就清零重试计数
            except Exception:
                retry_count += 1
                if retry_count > 3:  # 连续重试 3 次网络依然断开才退出
                    return "❌ 网络波动严重，检索大模型回答中断，请重新提交。"

        if status == "completed":
            msg_url = f"https://api.coze.cn/v3/chat/message/list?chat_id={chat_id}&conversation_id={conversation_id}"
            msg_res = requests.get(msg_url, headers=headers)
            for msg in msg_res.json().get("data", []):
                if msg["role"] == "assistant" and msg["type"] == "answer":
                    return msg["content"]
            return "⚠️ 大模型未返回 answer 字段内容。"
        else:
            return f"⚠️ 诊断中断，当前状态: {status}"
    except Exception as e:
        return f"❌ 通信异常: {str(e)}"

def create_radar_chart(metrics):
    categories = ['手势活跃度', '身体平稳度', '站姿挺拔度', '互动面向度']
    values = [
        metrics['gesture_score'],
        metrics['stability_score'],
        metrics['posture_score'],
        metrics['interaction_score']
    ]
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color='#1f77b4', width=2),
        name='教态得分'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[50, 100], tickfont=dict(size=10))
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=350
    )
    return fig

@st.cache_resource
def get_pose_tracker():
    return PoseTracker()

st.set_page_config(page_title="3D说课智能协同诊评系统", layout="wide")
st.title("🎓 3D说课智能协同诊评系统")
st.info("💡 平台已开启【三位一体多模态通道】：支持说课稿、PPT 课件与实况视频协同诊评。")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.write("### 📥 第一步：输入学生信息与上传素材")
    
    excel_custom_name = st.text_input(
        "📁 数据库归档表名称：",
        value="shuoke_records.xlsx",
        help="可自定义为如：2024级科教1班.xlsx",
    )

    if st.button("📂 一键在电脑中打开此 Excel 表格", use_container_width=True):
        target_file = (
            excel_custom_name
            if excel_custom_name.endswith(".xlsx")
            else excel_custom_name + ".xlsx"
        )
        if os.path.exists(target_file):
            os.startfile(os.path.abspath(target_file))
            st.toast(f"✅ 正在为您打开：{target_file}")
        else:
            st.warning("⚠️ 表格尚未创建，完成一次诊评后将自动生成！")
    student_name = st.text_input("👤 学生姓名：", placeholder="请输入进行说课的学生姓名")
    user_script = st.text_area("📝 请在此粘贴说课稿内容：", height=180, placeholder="例如：各位评委老师好...")
    
    uploaded_ppt = st.file_uploader("📊 上传说课 PPT 课件（支持 pptx, ppt, pdf 格式）：", type=["pptx", "ppt", "pdf"])
    uploaded_video = st.file_uploader("📹 上传说课实况视频（支持 mp4 格式）：", type=["mp4"])
    st.caption("💡 拍摄规范：建议横屏 16:9 录制，机位正对讲台，确保老师保持中景（胸部以上或全身可见）。")

    start_btn = st.button("🚀 开启 3D 三位一体智能协同诊评", use_container_width=True)

with col2:
    st.write("### 📑 第二步：多模态智能协同诊评报告")

    if start_btn:
        if not student_name.strip():
            st.warning("⚠️ 请先输入学生姓名，以便系统归档！")
        elif not user_script.strip():
            st.warning("⚠️ 请先输入说课稿内容！")
        else:
            script_report = ""
            ppt_report = ""
            ppt_name = ""
            posture_metrics = None
            processed_video_path = None

            with st.spinner("🔍 AI 正在进行多模态协同诊评，请稍候..."):
                script_report = call_coze_agent(COZE_API_KEY, COZE_SCRIPT_BOT_ID, user_script)

                if uploaded_ppt is not None:
                    ppt_name = uploaded_ppt.name
                    coze_file_id, upload_err = upload_file_to_coze(COZE_API_KEY, uploaded_ppt)
                    if coze_file_id:
                        ppt_report = call_coze_agent(COZE_API_KEY, COZE_PPT_BOT_ID, "请根据课件提供详细评测建议", coze_file_id)
                    else:
                        ppt_report = f"❌ 课件上传失败: {upload_err}"
                else:
                    ppt_report = "💡 本次实训未上传 PPT 课件。"

                if uploaded_video is not None:
                    temp_input_path = "temp_input.mp4"
                    processed_video_path = "temp_output.mp4"
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_video.read())

                    tracker = get_pose_tracker()
                    posture_metrics = tracker.process_video(
                        input_video=temp_input_path,
                        output_video=processed_video_path
                    )
                else:
                    posture_metrics = None

                save_to_excel(excel_custom_name, student_name, user_script, script_report, ppt_name, ppt_report, posture_metrics)

            st.success(f"✅ 三位一体协同诊评处理完毕！数据已归档至 【{excel_custom_name}】。学生：{student_name}")

            tab1, tab2, tab3 = st.tabs([
                "📝 说课稿诊断报告",
                "📊 PPT课件评估报告",
                "🧍 教态与肢体动作评估"
            ])

            with tab1:
                st.markdown(script_report)

            with tab2:
                st.markdown(ppt_report)

            with tab3:
                if posture_metrics:
                    # 🌟 熔断分支判断
                    if posture_metrics.get("is_valid", True):
                        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                        kpi1.metric("⭐ 教态综合总分", f"{posture_metrics['overall_score']} 分")
                        kpi2.metric("手势活跃度", f"{posture_metrics['gesture_score']} 分")
                        kpi3.metric("身体平稳度", f"{posture_metrics['stability_score']} 分")
                        kpi4.metric("站姿挺拔度", f"{posture_metrics['posture_score']} 分")
                        kpi5.metric("互动面向度", f"{posture_metrics['interaction_score']} 分")

                        col_chart, col_video = st.columns([1, 1.2])
                        with col_chart:
                            st.write("#### 🎯 教态能力雷达图")
                            fig = create_radar_chart(posture_metrics)
                            st.plotly_chart(fig, use_container_width=True)

                        with col_video:
                            st.write("#### 📹 骨骼动作追踪回放")
                            if os.path.exists(processed_video_path):
                                st.video(processed_video_path)

                        st.info(f"📋 **教态量化诊断评语**：\n\n{posture_metrics['report']}")
                    else:
                        st.error(posture_metrics["report"])
                        st.write("#### 📹 录像质检回放（供复盘排查）")
                        if os.path.exists(processed_video_path):
                            st.video(processed_video_path)
                else:
                    st.info("💡 本次诊断未上传说课视频。")
    else:
        st.write("👈 请在左侧上传说课稿、课件及实况视频，点击按钮开启协同诊评。")
