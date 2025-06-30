import gradio as gr
import time

# 定义所有步骤及其初始状态
steps_config = [
    {"name": "Step 1: 意图解析", "status": "🟡 Pending", "output": ""},
    {"name": "Step 2: 数据查询", "status": "🟡 Pending", "output": ""},
    {"name": "Step 3: 结果生成", "status": "🟡 Pending", "output": ""}
]

def run_agent(input_text):
    # 实时更新函数
    def update_step(step_idx, status, output=""):
        steps_config[step_idx]["status"] = status
        steps_config[step_idx]["output"] = output
        return gr.update(value=build_step_html())  # 刷新前端显示

    # 生成步骤的HTML显示
    def build_step_html():
        html = "<div style='font-family: sans-serif; padding: 20px; border: 1px solid #eee; border-radius: 8px;'>"
        for step in steps_config:
            html += f"""
            <div style='margin-bottom: 15px;'>
                <h4>{step['name']} <span style='color: {'green' if '✅' in step['status'] else 'orange'}'>
                    {step['status']}</span></h4>
                <div style='margin-left: 20px; color: #666'>{step['output']}</div>
            </div>
            """
        return html + "</div>"

    # 模拟步骤执行
    for i in range(len(steps_config)):
        time.sleep(1)  # 模拟处理耗时
        yield update_step(i, "🟢 Running...")
        
        # 实际处理逻辑（示例）
        if i == 0:
            yield update_step(i, "✅ Done", "识别出用户需要查询天气")
        elif i == 1:
            yield update_step(i, "✅ Done", "从API获取北京今日气温: 25°C")
        else:
            yield update_step(i, "✅ Done", "生成回复: 北京今天晴，25°C")

with gr.Blocks(title="多步骤Agent") as demo:
    gr.Markdown("## 任务执行进度")
    
    # 固定显示步骤面板
    steps_display = gr.HTML(
        value="<div>初始化中...</div>",
        elem_id="steps-panel"
    )
    
    # 输入和触发
    with gr.Row():
        input_text = gr.Textbox(label="输入指令", placeholder="请输入你的问题...")
        run_btn = gr.Button("执行", variant="primary")
    
    # 绑定事件
    run_btn.click(
        fn=run_agent,
        inputs=input_text,
        outputs=steps_display
    )

demo.launch(share=True)