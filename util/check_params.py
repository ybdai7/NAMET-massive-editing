printed_blocks = set()  # 用于记录已经打印的 TransformerBlock
first_transformer_block_printed = False

def print_model_structure(model, indent=0):
    global first_transformer_block_printed
    layer_name = model.__class__.__name__

    # 打印其他层名称和参数量
    param_count = sum(p.numel() for p in model.parameters())  # 计算参数总量
    print(" " * indent + f"Layer Name: {layer_name} - Parameters: {param_count}")

    # 打印参数形状
    if layer_name in ['ParallelEmbedding', 'ColumnParallelLinear', 'RowParallelLinear', 'RMSNorm']:
        param_shapes = [p.shape for p in model.parameters()]
        shape_info = ", ".join([f"{shape}" for shape in param_shapes])
        if param_shapes:
            print(" " * indent + f"  Parameter Shapes: [{shape_info}]")

    # 处理 TransformerBlock
    if layer_name == "TransformerBlock":
        if not first_transformer_block_printed:
            printed_blocks.add(model)  # 记录已打印的实例
            first_transformer_block_printed = True  # 标记为已打印

            # 打印子结构
            for name, layer in model.named_children():
                print(" " * (indent + 2) + f"Child Layer: {name} - Layer Type: {layer.__class__.__name__}")
                print_model_structure(layer, indent + 4)
        else:
            print(" " * indent + f"Child Layer ID: {id(model)}")  # 仅打印 ID
        return

    # 遍历子层并打印
    for name, layer in model.named_children():
        print(" " * (indent + 2) + f"Child Layer: {name} - Layer Type: {layer.__class__.__name__}")
        print_model_structure(layer, indent + 4)