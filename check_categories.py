import json

# 读取文件
with open('labels_my-project-name_2025-11-07-11-34-22.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计category_id使用情况
category_counts = {}
for ann in data.get('annotations', []):
    cat_id = ann.get('category_id')
    category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

print('当前category_id使用情况:')
for cat_id, count in sorted(category_counts.items()):
    print(f'category_id {cat_id}: {count}个标注')

print()
print(f'图片数量: {len(data.get("images", []))}')
print(f'标注数量: {len(data.get("annotations", []))}')

# 检查是否有categories字段
if 'categories' in data:
    print('\n文件已包含categories定义:')
    for cat in data['categories']:
        print(f"id={cat['id']}: {cat['name']}")
else:
    print('\n文件缺少categories字段定义')