# Test file for testing functions in utils
import os
import sys
import torch
from megatron import utils

def test_get_sequence_and_position_ids():
    eps = 1e-7
    
    # Case 1: 1 sequence
    tokens = torch.tensor([[1, 2, 3]])
    sequence_ids, position_ids = utils.get_sequence_and_position_ids(tokens, eos_token_id=3, bos_token_id=None)
    assert torch.all(torch.abs(sequence_ids - torch.tensor([[0, 0, 0]])) < eps)
    assert torch.all(torch.abs(position_ids - torch.tensor([[0, 1, 0]])) < eps)

    # Case 1: 2 sequence
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    sequence_ids, position_ids = utils.get_sequence_and_position_ids(tokens, eos_token_id=3, bos_token_id=None)
    assert torch.all(torch.abs(sequence_ids - torch.tensor([[0, 0, 0, 1, 1, 1]])) < eps)
    assert torch.all(torch.abs(position_ids - torch.tensor([[0, 1, 0, 1, 2, 3]])) < eps)

    # Case 1: multiple samples
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 1, 3, 1, 3, 4]])
    sequence_ids, position_ids = utils.get_sequence_and_position_ids(tokens, eos_token_id=3, bos_token_id=None)
    assert torch.all(torch.abs(sequence_ids - torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 2]])) < eps)
    assert torch.all(torch.abs(position_ids - torch.tensor([[0, 1, 0, 1, 2, 3], [0, 1, 0, 1, 0, 1]])) < eps)
    
def test_get_shifted_multimodal_position_ids():
    eps = 1e-7
    
    # Case 1: 1 image in between text
    text_positions = torch.tensor([[0, 2, 3, 4, 5, 6, 7]])
    vision_positions = torch.tensor([[1]])
    vision_seq_len = 2
    input_info = {
        "text": {
            "positions" : text_positions,
            "seq_length": 1
        },
        "vision": {
            "positions" : vision_positions,
            "seq_length": vision_seq_len,
        }
    }


    new_text_positions, new_vision_positions, _ = utils.get_shifted_multimodal_position_ids(input_info=input_info)
    assert torch.all(torch.abs(new_text_positions - torch.tensor([[0, 3, 4, 5, 6, 7, 8]])) < eps)
    assert torch.all(torch.abs(new_vision_positions - torch.tensor([[1, 2]])) < eps)
    
    # Case 2: Different image position
    text_positions = torch.tensor([[0, 1, 2, 5]])
    vision_positions = torch.tensor([[3]])
    vision_seq_len = 3
    input_info = {
        "text": {
            "positions" : text_positions,
            "seq_length": 1
        },
        "vision": {
            "positions" : vision_positions,
            "seq_length": vision_seq_len,
        }
    }

    new_text_positions, new_vision_positions, new_audio_positions = utils.get_shifted_multimodal_position_ids(input_info=input_info)

    assert torch.all(torch.abs(new_text_positions - torch.tensor([[0, 1, 2, 7]])) < eps)
    assert torch.all(torch.abs(new_vision_positions - torch.tensor([[3, 4, 5]])) < eps)

    # Case 3: Image position in the start 
    text_positions = torch.tensor([[1, 2, 5]])
    vision_positions = torch.tensor([[0]])
    vision_seq_len = 3
    input_info = {
        "text": {
            "positions" : text_positions,
            "seq_length": 1
        },
        "vision": {
            "positions" : vision_positions,
            "seq_length": vision_seq_len,
        }
    }

    new_text_positions, new_vision_positions, new_audio_positions = utils.get_shifted_multimodal_position_ids(input_info=input_info)

    assert torch.all(torch.abs(new_text_positions - torch.tensor([[3, 4, 7]])) < eps)
    assert torch.all(torch.abs(new_vision_positions - torch.tensor([[0, 1, 2]])) < eps)

    # Case 4: Image position at the end
    text_positions = torch.tensor([[0, 1, 2]])
    vision_positions = torch.tensor([[3]])
    vision_seq_len = 3
    input_info = {
        "text": {
            "positions" : text_positions,
            "seq_length": 1
        },
        "vision": {
            "positions" : vision_positions,
            "seq_length": vision_seq_len,
        }
    }

    new_text_positions, new_vision_positions, new_audio_positions = utils.get_shifted_multimodal_position_ids(input_info=input_info)

    assert torch.all(torch.abs(new_text_positions - torch.tensor([[0, 1, 2]])) < eps)
    assert torch.all(torch.abs(new_vision_positions - torch.tensor([[3, 4, 5]])) < eps)

    # Case 5: Multiple images 
    text_positions = torch.tensor([[0, 2, 5]])
    vision_positions = torch.tensor([[1, 4, ]])
    vision_seq_len = 3
    input_info = {
        "text": {
            "positions" : text_positions,
            "seq_length": 1
        },
        "vision": {
            "positions" : vision_positions,
            "seq_length": vision_seq_len,
        }
    }

    new_text_positions, new_vision_positions, new_audio_positions = utils.get_shifted_multimodal_position_ids(input_info=input_info)

    assert torch.all(torch.abs(new_text_positions - torch.tensor([[0, 4, 9]])) < eps)
    assert torch.all(torch.abs(new_vision_positions - torch.tensor([[1, 2, 3, 6, 7, 8]])) < eps)

    # Case 6: Zero images 
    text_positions = torch.tensor([[0, 1, 2, 5]])
    vision_positions = torch.tensor([[]])
    vision_seq_len = 3
    input_info = {
        "text": {
            "positions" : text_positions,
            "seq_length": 1
        },
        "vision": {
            "positions" : vision_positions,
            "seq_length": vision_seq_len,
        }
    }

    new_text_positions, new_vision_positions, new_audio_positions = utils.get_shifted_multimodal_position_ids(input_info=input_info)

    assert torch.all(torch.abs(new_text_positions - torch.tensor([[0, 1, 2, 5]])) < eps)
    assert torch.all(torch.abs(new_vision_positions - torch.tensor([[]])) < eps)

    # Case 7: Zero text
    text_positions = torch.tensor([[]])
    vision_positions = torch.tensor([[3]])
    vision_seq_len = 3
    input_info = {
        "text": {
            "positions" : text_positions,
            "seq_length": 1
        },
        "vision": {
            "positions" : vision_positions,
            "seq_length": vision_seq_len,
        }
    }

    new_text_positions, new_vision_positions, new_audio_positions = utils.get_shifted_multimodal_position_ids(input_info=input_info)

    assert torch.all(torch.abs(new_text_positions - torch.tensor([[]])) < eps)
    assert torch.all(torch.abs(new_vision_positions - torch.tensor([[3, 4, 5]])) < eps)

    # Case 8: Multiple samples 
    text_positions = torch.tensor([[0, 2, 5, -1, -1], [0, 1, 2, 5, -1]])
    vision_positions = torch.tensor([[1, 4, -1], [3, -1, -1]])
    vision_seq_len = 3
    input_info = {
        "text": {
            "positions" : text_positions,
            "seq_length": 1
        },
        "vision": {
            "positions" : vision_positions,
            "seq_length": vision_seq_len,
        }
    }

    new_text_positions, new_vision_positions, new_audio_positions = utils.get_shifted_multimodal_position_ids(input_info=input_info)
    assert torch.all(torch.abs(new_text_positions - torch.tensor([[0, 4, 9, 10, 11], [0, 1, 2, 7, 8]])) < eps)
    assert torch.all(torch.abs(new_vision_positions - torch.tensor([[1, 2, 3, 6, 7, 8, 12, 13, 14], [3, 4, 5, 9, 10, 11, 12, 13, 14]])) < eps)

def test_get_proxy_tokens():
    eps = 1e-7
    
    # Case 0: 0 images # TODO
    vision_positions = torch.tensor([[-1]])
    vision_seq_len = 1
    proxy_vision_tokens = utils.get_proxy_tokens(position_ids=vision_positions, seq_length=vision_seq_len, pad_id=0)
    assert torch.all(torch.abs(proxy_vision_tokens - torch.tensor([[0]])) < eps)

    # Case 1: 1 image
    vision_positions = torch.tensor([[1]])
    vision_seq_len = 1
    proxy_vision_tokens = utils.get_proxy_tokens(position_ids=vision_positions, seq_length=vision_seq_len, pad_id=0)
    assert torch.all(torch.abs(proxy_vision_tokens - torch.tensor([[-100]])) < eps)

    # Case 2: multiple images 
    vision_positions = torch.tensor([[1, 2]])
    vision_seq_len = 1
    proxy_vision_tokens = utils.get_proxy_tokens(position_ids=vision_positions, seq_length=vision_seq_len, pad_id=0)
    assert torch.all(torch.abs(proxy_vision_tokens - torch.tensor([[-100, -101]])) < eps)

    # Case 3: multiple images, seq len > 1
    vision_positions = torch.tensor([[3, 5]])
    vision_seq_len = 3
    proxy_vision_tokens = utils.get_proxy_tokens(position_ids=vision_positions, seq_length=vision_seq_len, pad_id=0)
    assert torch.all(torch.abs(proxy_vision_tokens - torch.tensor([[-100, -100, -100, -101, -101, -101]])) < eps)

    # Case 4: multiple samples with padding
    vision_positions = torch.tensor([[1, 2], [3, -1]])
    vision_seq_len = 2
    proxy_vision_tokens = utils.get_proxy_tokens(position_ids=vision_positions, seq_length=vision_seq_len, pad_id=0)
    assert torch.all(torch.abs(proxy_vision_tokens - torch.tensor([[-100, -100, -101, -101], [-100, -100, 0, 0]])) < eps)

def test_get_multimodal_mask():
    
    # Case 0: 0 images
    tokens = torch.tensor([[2, 2]])
    multimodal_mask = utils.get_multimodal_mask(tokens, text_pad_id=0)
    correct_mask = torch.tensor([
        [
            [False, False],
            [False, False]
        ]
    ])
    assert torch.all(multimodal_mask == correct_mask)
    
    # Case 1: 1 image
    tokens = torch.tensor([[-100]])
    multimodal_mask = utils.get_multimodal_mask(tokens, text_pad_id=0)
    assert torch.all(multimodal_mask == torch.tensor([[[True]]]))

    # Case 2: 2 image with multiple tokens per image
    tokens = torch.tensor([[-100, -100, -101, -101]])
    multimodal_mask = utils.get_multimodal_mask(tokens, text_pad_id=0)
    correct_mask = torch.tensor([
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True]
        ]
    ])
    assert torch.all(multimodal_mask == correct_mask)

    # Case 2: Multiple images with multiple tokens per image and text
    tokens = torch.tensor([[0, -100, -100, 1, -101, -101]])
    multimodal_mask = utils.get_multimodal_mask(tokens, text_pad_id=0)
    correct_mask = torch.tensor([
        [
            [False, False, False, False, False, False],
            [False, True, True, False, False, False],
            [False, True, True, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, True, True],
            [False, False, False, False, True, True],
        ]
    ])
    assert torch.all(multimodal_mask == correct_mask)

    # Case 3: Multiple samples with text, images, and padding
    tokens = torch.tensor([[0, -100, -100, 1, -101, -101], [3, -100, -100, -101, -101, 0]])
    multimodal_mask = utils.get_multimodal_mask(tokens, text_pad_id=0)
    correct_mask = torch.tensor([
        [
            [False, False, False, False, False, False],
            [False, True, True, False, False, False],
            [False, True, True, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, True, True],
            [False, False, False, False, True, True],
        ], 
        [
            [False, False, False, False, False, False],
            [False, True, True, False, False, False],
            [False, True, True, False, False, False],
            [False, False, False, True, True, False],
            [False, False, False, True, True, False],
            [False, False, False, False, False, False],
        ]
    ])
    assert torch.all(multimodal_mask == correct_mask)

def test_get_multimodal_attn_mask():
    eps = 1e-7

    text_tokens = torch.tensor([[2, 3, 1, 5, 9], [2, 3, 9, 1, 0]])
    text_positions = torch.tensor([[0, 2, 3, 5, 6], [0, 1, 2, 4, -1]])
    vision_positions = torch.tensor([[1, 4, -1], [3, -1, -1]])
    vision_seq_len = 3
    input_info = {
        "text": {
            "input": text_tokens,
            "positions" : text_positions,
            "seq_length": 1
        },
        "vision": {
            "positions" : vision_positions,
            "seq_length": vision_seq_len,
        }
    }

    labels=text_positions
    input_seq_length=11
    eod_token=1
    pad_token=0
    bos_token=None
    concat_data=True
    attn_uses_sequence_id=True
    batch_size = labels.shape[0]
    label_length = labels.shape[1]

    shifted_text_positions, shifted_vision_positions, shited_audio_positions = utils.get_shifted_multimodal_position_ids(input_info, position_pad_id=-1)
    shifted_multimodal_position_ids = torch.cat((shifted_text_positions, shifted_vision_positions), dim=1)
    attention_mask, position_ids = utils.get_multimodal_attn_mask(
        text_tokens=input_info["text"]["input"],
        vision_positions=input_info["vision"]["positions"],
        audio_positions=None,
        vision_seq_length=input_info["vision"]["seq_length"],
        input_seq_length=input_seq_length,
        shifted_multimodal_position_ids=shifted_multimodal_position_ids,
        eos_token_id=eod_token,
        bos_token_id=bos_token,
        position_pad_token_id=-1, # TODO, get whatever is used in streaming
        text_pad_token_id=pad_token,
        concat_data=concat_data,
        attn_uses_sequence_id=attn_uses_sequence_id,
        device=labels.device,
    )

    '''
    text_tokens = torch.tensor([[2, 3, 1, 5, 9], [2, 3, 9, 1, 0]])
    
    text_positions = torch.tensor([[0, 2, 3, 5, 6], [0, 1, 2, 4, -1]])
    vision_positions = torch.tensor([[1, 4, -1], [3, -1, -1]])
    '''
    correct_mask = torch.tensor([
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
         [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]],

        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]]])
    correct_mask = correct_mask.view(batch_size, 1, input_seq_length, input_seq_length)
    correct_mask = correct_mask < 0.5
    assert torch.all(attention_mask == correct_mask)

    correct_positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]], dtype=torch.int32)
    assert torch.all(position_ids == correct_positions)


# Main function
def main():
    test_get_sequence_and_position_ids()
    test_get_shifted_multimodal_position_ids()
    test_get_proxy_tokens()
    test_get_multimodal_mask()
    test_get_multimodal_attn_mask()

if __name__ == "__main__":
    main()