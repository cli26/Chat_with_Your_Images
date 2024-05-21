# Chat_with_Your_Images

通过Azure OpenAI 访问 GPT4 Turbo with Vision。建立包含图片数据的知识库，并基于 REST API 与这个知识库进行问答互动。在这个 repo 中以零售场景中拍摄的货架图片为例，首先通过 Florence model 对图片中的商品进行分割，再通过 GPT4 Turbo with Vision 对单个商品图片进行分析，可实现缺货提醒，排面检查等一系列后续任务。

## Prerequisites
1. An Azure subscription.
2. Access granted to Azure OpenAI in the desired Azure subscription. You can apply for access to Azure OpenAI by completing the form at https://aka.ms/oai/access.
3. An Azure OpenAI resource with the GPT-4 Turbo with Vision model deployed.
4. Be sure that you're assigned at least the Cognitive Services Contributor role for the Azure OpenAI resource.

## Pricing
通过如下代码逻辑，根据图片的大小计算token，从而得到一个成本预计。

```
import numpy as np

def calculate_image_tokens(width, height):
    # Determine the long and short edges
    long_edge = max(width, height)
    short_edge = min(width, height)

    # Resize rules
    if long_edge > 2048:
        # Resize long edge to 2048 and scale short edge proportionally
        scale_factor = 2048.0 / long_edge
        short_edge_scaled = short_edge * scale_factor
        # Resize short edge to 768 and scale long edge (now 2048) proportionally
        if short_edge_scaled > 768:
            scale_factor = 768.0 / short_edge_scaled
            long_edge_final = 2048 * scale_factor
            short_edge_final = 768
        else:
            long_edge_final = 2048
            short_edge_final = short_edge_scaled
    elif long_edge <= 2048 and short_edge > 768:
        # Resize short edge to 768 and scale long edge proportionally
        scale_factor = 768.0 / short_edge
        long_edge_final = long_edge * scale_factor
        short_edge_final = 768
    else:
        # No resizing needed
        long_edge_final = long_edge
        short_edge_final = short_edge
        
    print('long_edge_final: ', long_edge_final)
    print('short_edge_final: ', short_edge_final)

    # Calculate the number of 512x512 tiles needed to cover the resized image
    tiles_across = int(np.ceil(long_edge_final / 512))
    tiles_down = int(np.ceil(short_edge_final / 512))

    # Total number of tiles needed
    n_tiles = tiles_across * tiles_down

    # Calculate the token count for the image
    image_tokens = 85 + 170 * n_tiles

    return image_tokens

# Test the function with a sample image size
sample_width = int(input('Enter the image width: '))
sample_height = int(input('Enter the image height: '))

# Calculate the tokens for the sample image size
tokens = calculate_image_tokens(sample_width, sample_height)
price = tokens/1000*0.01

print('tokens: ', tokens)
print('price: %.6f $' % price)
```
