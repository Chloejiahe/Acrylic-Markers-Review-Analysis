import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re

# --- 1. 核心词库配置 (Feature Keywords) ---
# 这里只放两个示例，你可以把你之前整理的 8 大板块全部填进去
FEATURE_DIC = {
     '颜色种类': {
            '正面-色彩丰富': ['many colors', 'lot of colors', 'plenty of colors', 'good range', 'great variety', 'great selection', 'every color', 'all the colors', 'so many options'],
            '负面-色彩单调/反馈': ['limited range', 'not enough colors', 'wish for more', 'missing colors', 'disappointed with selection', 'needs more colors'],
            '正面-套装/数量选择满意': ['love the large set', 'great number of colors', 'perfect amount of colors', 'huge set of 72', 'full set is amazing', 'good assortment'],
            '负面-套装/数量选择不满意': ['wish for a smaller set', 'too many colors', 'no smaller option', 'forced to buy the large set', 'have to buy the whole set'],
            '正面-色系规划满意': ['great color selection', 'perfect pastel set', 'good range of skin tones', 'well-curated palette', 'love the color story', 'beautiful assortment of colors', 'has every color I need'],
            '负面-色系规划不满': ['missing key colors', 'no true red', 'needs more grays', 'too many similar colors', 'palette is not useful', 'wish it had more pastels', 'poor color selection', 'needs more skin tones'],
            '中性-提及色彩丰富度': ['color range', 'color variety', 'color selection', 'number of colors', 'range of shades','selection of hues', 'color assortment', 'color palette', 'spectrum of colors', 'array of colors', 'how many colors'],

            },
        '色彩一致性': {
            '正面-颜色准确': ['true to color', 'match the cap', 'accurate color', 'color accuracy', 'exact color', 'matches perfectly', 'consistent color', 'consistency'],
            '负面-颜色偏差': ['inconsistent', 'different shade', 'not the same', 'misleading cap', 'cap is wrong', 'color is off', 'darker than cap', 'lighter than cap', "doesn't match", 'wrong color'],
            '正面-设计-颜色准确 (VS 笔帽)': ['true to color', 'match the cap', 'matches the cap perfectly', 'cap is a perfect match', 'cap is accurate'],
            '负面-设计-颜色误导 (VS 笔帽)': ['misleading cap', 'cap is wrong', 'cap is a lie', "color doesn't match the barrel", 'the cap color is way off', 'nothing like the cap'],
            '正面-营销-颜色准确(VS 网图)': ['exactly as advertised', 'what you see is what you get', 'matches the online photo', 'true to the swatch', 'photo is accurate'],
            '负面-营销-图片误导 (VS 网图)': ['looks different from the online swatch', 'not the color in the picture', 'misrepresented color', 'photo is misleading', 'swatch card is inaccurate'],
            '正面-生产-品控(VS 其他笔)': ['consistent color', 'consistency', 'no variation between pens', 'reliable color', 'batch is consistent'],
            '负面-生产-品控偏差(VS 其他笔)': ['inconsistent batch', 'color varies from pen to pen', 'my new pen is a different shade', 'no quality control', 'batch variation'],
            '中性-提及颜色准确性': ['color accuracy', 'true to color', 'accurate color', 'color match', 'exact shade','color representation', 'color consistency', 'pen to pen consistency'],
            '中性-提及笔帽颜色': ['cap color', 'barrel color', 'match the cap', 'color of the cap', 'color on the barrel','cap match', 'indicator on the cap', 'swatch on the barrel', 'color indicated on the pen'],
            '中性-提及网图/色卡': ['swatch card', 'color swatch', 'online swatch', 'swatching', 'swatch test', 'product photo', 'online photo', 'listing photo', 'website image', 'advertised picture', 'photo in the listing'],

            },
        '色彩饱和度与混合': {
            '正面-鲜艳/饱和': ['bright colors', 'nice and bright', 'beautifully bright', 'richly saturated', 'perfectly saturated', 'deeply saturated', 'nice saturation', 'vibrant colors', 'rich colors', 'colors pop'],
            '负面-太鲜艳/刺眼': ['garish colors', 'colors are too loud', 'too neon', 'too bright', 'too fluorescent', 'overly bright'],
            '负面-暗淡/褪色': ['dull', 'faded', 'pale', 'washed out', 'not bright', 'too pale', 'lackluster', 'colors are too dull', 'muddy colors', 'colors look dirty', 'desaturated','doesn\'t show up well',],
            '正面-遮盖力强': ['great coverage', 'completely opaque', 'one coat covers all', 'thick paint feel', 'works on rocks', 'shows up on black', 'vibrant on dark surfaces', 'solid color', 'creamy texture'],
            '负面-遮盖力差': ['too transparent', 'very sheer', 'watery', 'streaky', 'takes many coats', 'see through', 'thin ink', 'runny', 'faint colors'],
            '中性-提及饱和度': ['saturation', 'vibrancy', 'color intensity', 'richness of color', 'color depth', 'pigment load', 'high saturation', 'low saturation', 'deep saturation'],
            '正面-叠色顺滑': ['layers well', 'easy to layer', 'good layering', 'smooth layering', 'layers perfectly', 'buildable color', 'smooth finish', 'not streaky', 'even layers', 'consistent texture', 'creamy application', 'doesn\'t disturb bottom layer', 'sits on top nicely', 'doesn\'t lift previous paint', 'no bleeding between layers'],
            '负面-叠色困难': ['takes forever to dry before layering', 'smears the bottom color', 'lifts if not 100% dry', 'lifts the layer underneath', 'scratches the paint off', 'rubs off previous layer', 'strips the paint', 'disturbs dried paint', 'tears up the bottom color', 'too transparent', 'very sheer', 'streaky when layering', 'can see the color underneath', 'takes too many coats to cover', 'patchy coverage' ],

        },
        '色系评价': {
            '正面-喜欢标准/基础色系': ['good standard colors', 'love the basic set', 'has all the primary colors', 'classic colors',  'great essential colors', 'perfect starter palette', 'all the fundamental colors',
                          'love the traditional colors', 'a solid basic palette', 'includes primary and secondary colors','just the basics I needed', 'don\'t need anything fancy'],
            '正面-喜欢鲜艳/饱和色系': ['love the vibrant colors', 'love the rich colors', 'love the bold colors', 'love how vivid the colors are','highly saturated', 'nicely saturated', 'colors are saturated', 'great saturation', 'amazing saturation', 'deeply saturated',
                             'colors pop', 'really pop', 'makes the colors pop', 'colors really stand out', 'colors jump off the page','not dull at all', 'anything but muted', 'so full of life'],
            '正面-喜欢粉彩色/柔和系': ['love the pastel colors', 'beautiful pastels', 'adore the soft colors', 'perfect muted tones','calming color palette', 'soothing shades', 'gentle on the eyes', 'doesn\'t hurt my eyes',
                          'subtle and elegant', 'love the macaron colors', 'the muted palette is gorgeous','unlike neon', 'unlike fluorescent', 'not overwhelming'],
            '正面-喜欢复古/怀旧色系': ['love the vintage colors', 'perfect retro palette', 'nostalgic color scheme', 'love the old school colors','adore the vintage feel', 'love the retro vibe', 'perfect nostalgic feel',
                          'great aged palette', 'beautiful antique colors', 'love the heritage colors','70s color palette', 'mid-century modern colors','love the mustard yellow', 'love the avocado green', 'love the burnt orange'],
            '正面-喜欢莫兰迪色系': ['love the morandi colors', 'adore the morandi palette', 'perfect morandi palette', 'beautifully dusty colors', 'love the grayish tones', 'muted and elegant', 'sophisticated colors',
                        'calming color palette', 'soothing to the eye', 'understated and beautiful','looks so high-end', 'elegant color scheme', 'love the muted aesthetic'],
            '正面-喜欢中性/肤色系': ['love the skin tones', 'great range of skin tones', 'perfect for portraits', 'excellent for portraiture','realistic skin tones', 'beautiful flesh tones', 'wide variety of skin colors',
                         'perfect neutral palette', 'love the neutral colors', 'great selection of neutrals', 'beautiful earth tones','adore the earthy palette', 'good selection of beiges', 'love the taupes'],
            '正面-喜欢大地/自然色系': ['love the earth tones', 'adore the earthy palette', 'beautiful natural colors', 'gorgeous nature-inspired colors', 'love the botanical colors', 'perfect botanical palette', 'beautiful forest greens', 'love the sage green',
                          'stunning desert tones', 'love the terracotta shades', 'beautiful clay tones','gorgeous ocean blues', 'love the coastal colors', 'calming mountain palette'],
            '正面-喜欢灰色系': ['love the gray scale', 'great set of cool grays', 'perfect warm grays', 'good neutral grays','excellent grayscale palette', 'adore the range of grays', 'beautiful selection of grays',
                       'love the different shades of gray', 'perfect for monochromatic work', 'great for shadows and shading','the warm grays are beautiful', 'the cool grays are perfect'],
            '正面-喜欢季节/主题色系': ['love the seasonal set', 'perfect seasonal palette', 'beautiful themed set', 'gorgeous forest colors', 'love the ocean tones', 'stunning coastal palette',
                          'perfect autumn palette', 'love the fall colors', 'beautiful autumn shades','vibrant spring colors set', 'love the spring palette', 'beautiful summer tones', 'perfect tropical palette',
                         'love the winter palette', 'gorgeous wintery shades','perfect for christmas', 'great holiday color scheme', 'love the halloween set'],
            '正面-喜欢霓虹/荧光色系': ['love the neon colors', 'beautiful neon colors', 'amazing fluorescent palette', 'super bright neon','the neon really pops', 'vibrant neon shades', 'glows under blacklight', 'perfect for blacklight art',
                          'electric colors are stunning', 'day-glo colors are so bright'],
            '正面-喜欢金属/珠光色系': ['love the metallic colors', 'great metallic effect', 'beautiful metallic sheen', 'shiny metal finish','gorgeous chrome finish', 'looks like real metal', 'love the pearlescent finish', 'beautiful shimmer',
                          'amazing liquid chrome effect', 'very reflective', 'stunning iridescent colors', 'love the lustre','the gold is so rich', 'the silver is brilliant', 'beautiful copper tone'],
            '负面-色系搭配不佳': ["palette is ugly", "colors don't go well together", 'weird color combination', 'unusable colors in set','poorly curated', 'terrible color choices', 'bad color selection', 'the colors clash',
                        'no color harmony', 'strange selection of colors', 'too many similar colors', 'missing key colors', 'not a cohesive palette', 'thoughtless color selection', 'colors are jarring'],
            '中性-提及标准/基础色系': ['standard colors', 'basic set', 'primary colors', 'secondary colors', 'classic colors',
                          'essential colors', 'core colors', 'fundamental palette', 'starter set of colors','introductory palette', 'traditional colors',
                          'basic color wheel'],
            '中性-提及鲜艳/饱和色系': ['vibrant colors', 'bright colors', 'bold colors', 'rich colors', 'vivid colors','color intensity', 'richness of color', 'deep colors','intense colors', 'highly saturated', 'brilliant colors', 'bold palette',
                          'strong colors', 'pop of color', 'jewel tones'],
            '中性-提及粉彩色/柔和系': ['pastel', 'pastels', 'pastel colors', 'pastel shades', 'soft colors', 'subtle shades', 'mild colors', 'macaron colors', 'muted tones',
                       'muted colors', 'gentle colors','delicate colors', 'light colors', 'pale palette', 'low saturation', 'desaturated colors','baby colors',
                       'baby pink', 'baby blue', 'mint green', 'lavender','sorbet colors', 'ice cream colors', 'easter colors'
            ],
            '中性-提及复古/怀旧色系': ['vintage colors', 'retro palette', 'nostalgic colors', 'old school colors', 'sepia tones','aged palette', 'antique colors', 'heritage colors', 'desaturated palette',
                           'mid-century modern colors', 'dusty rose', 'mustard yellow', 'avocado green'],
            '中性-提及莫兰迪色系': [ 'morandi', 'morandi colors', 'morandi palette', 'dusty colors', 'grayish tones', 'muted palette', 'hazy colors', 'understated colors', 'sophisticated palette',
                          'low saturation colors', 'dusty pink', 'sage green', 'pale blue', 'terracotta', 'beige tones'],
            '中性-提及中性/肤色系': ['skin tones', 'flesh tones', 'skin tone palette', 'portrait palette', 'range of skin tones',
                          'neutral palette', 'neutral colors', 'neutrals', 'set of neutrals', 'earth tones'],
            '中性-提及大地/自然色系': ['earth tones', 'earthy palette', 'natural colors', 'nature-inspired colors',  'botanical colors', 'botanical palette', 'forest greens', 'sage green', 'olive green', 'moss green',
                          'desert tones', 'canyon colors',  'clay tones','ocean blues', 'coastal colors', 'seafoam green', 'sky blue','mountain palette', 'stone grays', 'slate gray'],
            '中性-提及灰色系': ['gray scale', 'grayscale', 'grays', 'shades of gray', 'set of grays','cool grays', 'warm grays', 'neutral grays', 'french grey', 'payne\'s grey','light gray', 'dark gray', 'charcoal gray', 'slate gray', 'silver gray', 'monochromatic palette'],

            '中性-提及霓虹/荧光色系': ['neon colors', 'neon palette', 'neon set', 'fluorescent colors', 'fluorescent shades',
                          'highlighter colors', 'day-glo colors', 'electric colors', 'glow under blacklight', 'blacklight reactive', 'uv reactive ink'],
            '中性-提及金属/珠光色系': ['metallic ink', 'metallic colors', 'metallic finish', 'metallic sheen', 'liquid chrome', 'mirror finish','pearlescent effect', 'pearlescent finish', 'mother-of-pearl effect',
                          'shimmering ink', 'shimmer effect', 'glittering effect', 'lustre', 'iridescent','gold ink', 'silver ink', 'bronze ink', 'copper ink'],
            '中性-提及色系搭配': ['color palette', 'color combination', 'color scheme', 'color story', 'color assortment', 'curated palette', 'well-chosen colors', 'colors work together', 'color harmony',
                       'range of colors', 'selection of colors'],
        },
        '笔头表现': {
            '正面-双头设计认可': ['love the dual tip', 'love the two tips', 'love that it has two sides', 'love the dual nibs','great having two tips', 'useful dual tip', 'handy dual tip', 'convenient to have two tips',
                        'best of both worlds', 'love the brush and fine tip combo', 'perfect combination of tips','like having two pens in one', 'great for switching between broad and fine'],
            '负面-双头设计抱怨': ['useless dual tip', 'redundant dual tip', 'unnecessary dual tip', "don't need the dual tip", 'never use the other side', 'only use one side', 'the other end is useless',
                       'wish it was a single tip', 'wish they sold them separately', 'would rather have two separate pens','only bought it for the brush side', 'one of the tips is useless'],
            '正面-软头表现好': ['love the brush tip', 'great brush nib', 'smooth application with the brush', 'brush tip is very responsive','flexible brush tip', 'soft brush tip allows for variation','happy with the brush',],
            '负面-软头表现差': ['brush tip frays', 'brush tip split', 'brush tip wore out', 'brush tip lost its point','inconsistent brush line','brush tip clogged', 'ink won\'t flow from the brush'],
            '正面-细头表现好': ['love the fine tip', 'precise fine liner', 'crisp fine lines', 'excellent for fine details', 'perfect for writing in small spaces', 'great for intricate work','happy with the bullet','happy with the fine',
                      'allows for detailed drawing', 'perfect for outlining', 'creates super thin lines'],
            '负面-细头表现差': ['fine tip is scratchy','fine tip dried out', 'bent the fine tip', 'fine tip broke', 'inconsistent fine line','fine nib wore down', 'tip lost its point', 'fine tip feels fragile'],
            '正面-凿头表现好': ['sharp chisel edge', 'maintains a sharp edge', 'makes clean broad strokes',  'perfect for block lettering', 'great for filling large areas',
                      'can create both thick and thin lines', 'consistent broad lines', 'even coverage with broad side'],
            '负面-凿头表现差': ['chisel tip is too broad', 'chisel tip is too thick', 'chisel tip is too narrow','chisel tip wore down', 'loses its edge quickly', 'edge became rounded',
                      'dull chisel tip', 'dull chisel edge', 'can\'t get a sharp line', 'no longer has a crisp edge','inconsistent broad line', 'chisel tip crumbled', 'chisel tip chipped'],
            '正面-圆头表现好': ['sturdy bullet tip', 'consistent bullet nib', 'reliable bullet point', 'smooth bullet nib', 'perfect for uniform strokes', 'great for consistent line width', ],
            '负面-圆头表现差': ['bullet tip skips', 'inconsistent line from bullet tip','bullet nib is dry', 'bullet tip dried out', 'ink doesn\'t flow to the bullet tip', 'arrived with a dry bullet tip','wobbly bullet tip', 'loose bullet tip', 'bullet tip receded', 'bullet tip broke', 'bullet tip cracked', 'scratchy bullet nib', 'bullet tip scratches paper', 'blotchy line from bullet tip'],
            '正面-笔头弹性好/软硬适中': ['flexible', 'great flexibility', 'nice spring', 'good snap', 'bouncy tip', 'soft tips',],
            '正面-泵吸/按压表现好': ['easy to prime', 'fast to start', 'primed quickly', 'ink flowed almost instantly', 'ready to use in seconds', 'minimal pumping needed', 'quick to get the paint moving', 'smooth pump action', 'responsive valve', 'springs back perfectly', 'easy on the hands', 'gentle pressure required', 'consistent pumping mechanism', 'even ink flow after priming', 'controlled saturation', 'saturated the tip perfectly', 'no dry spots after starting', 'ink spreads evenly across the nib'],
            '负面-泵吸/按压表现差': ['hard to prime', 'took forever to start', 'pumping for 10 minutes', 'arm got tired shaking and pumping', 'struggled to get ink out', 'frustrating to start', 'wouldn\'t prime', 'impossible to get the paint down', 'nib pushed in and stayed', 'tip got stuck inside', 'valve is stuck', 'no spring back', 'nib receded into the barrel', 'pushed the tip all the way in', 'defective pump', 'ink gushed out', 'massive blob of paint', 'splattered everywhere', 'leaked during priming', 'way too much ink came out', 'uncontrollable flow', 'huge mess after pumping'],
            '负面-笔头过软/过硬/无弹性': ['too stiff', 'too firm', 'too soft', 'no flexibility', 'mushy', 'hard to control flex'],
            '正面-笔尖可替换': ['replaceable nibs', 'can replace the tips', 'interchangeable tips', 'love the replacement nibs'],
            '负面-笔尖不可替换': ["can't replace the nib", 'wish the tips were replaceable', 'no replacement nibs'],
            '正面-软头(Brush)-粗细变化好': ['good line variation', 'can make thick and thin lines', 'great control over stroke width', 'responsive brush'],
            '负面-软头(Brush)-粗细难控': ['hard to get a thin line', 'only makes thick strokes', 'inconsistent line width', 'no line variation'],
            '正面-细头(Fine)-粗细适合细节': ['perfect for details', 'love the fine tip', 'thin enough for writing', 'great for fine lines', 'super fine point'],
            '负面-细头(Fine)-粗细不合适': ['too thick for a fine liner', 'not a true fine', 'wish it was thinner', 'still too broad for small spaces'],
            '正面-凿头(Chisel)-宽度合适': ['perfect width for highlighting', 'good broad edge', 'nice thick lines for headers'],
            '负面-凿头(Chisel)-宽度不合适': ['too wide for my bible', 'too narrow for a highlighter', 'chisel tip is too thick'],
            '正面-圆头(Bullet)-粗细均匀': ['nice medium point', 'consistent line width', 'good for coloring', 'reliable bullet tip'],
            '负面-圆头(Bullet)-粗细问题': ['bullet tip is too bold', 'not a medium point as advertised'],
            '中性-提及双头设计': ['dual tip', 'dual-tip', 'two tips', 'two sides', 'double ended'],
            '中性-提及软头': ['brush tip', 'brush nib', 'soft tip', 'brush pen','brush tips'],
            '中性-提及细头': ['fine tip', 'fine liner', 'fine point', 'fineliner', 'extra fine'],
            '中性-提及凿头': ['chisel tip', 'chisel nib', 'broad tip', 'highlighter tip'],
            '中性-提及圆头': ['bullet tip', 'bullet nib', 'medium point', 'round tip'],
            '中性-提及弹性': ['tip flexibility', 'flexibility', 'stiffness', 'softness', 'hardness', 'spring back', 'tip snap'],
            '中性-提及替换笔尖': ['replaceable nibs', 'replacement nibs', 'interchangeable tips',],
            '中性-提及软头粗细': ['line variation', 'stroke width', 'thick and thin lines', 'line weight'],
            '中性-提及细头粗细': ['line width', 'point size', '0.3mm tip', '0.3mm nib', '0.3mm point', '0.4mm tip', '0.4mm nib', '0.4mm point', '0.5mm tip', '0.5mm nib', '0.5mm point','extra fine point'],
            '中性-提及凿头宽度': ['chisel width', 'broad edge', 'line thickness'],
            '中性-提及圆头粗细': ['medium point', 'consistent line width', 'boldness'],
            '中性-提及点状头': ['dot marker', 'dot pen', 'dot tip', 'dot markers', 'dot pens', 'bingo daubers', 'dauber pens', 'dot art'],
            '中性-提及点状头点型': ['dot shape', 'dot roundness', 'circle shape'],
            '中性-提及点状头尺寸': ['dot size', 'dot size variation', 'pressure sensitivity'],
        },
        '笔头耐用性': {
            '正面-耐磨损/抗分叉': ["doesn't fray", 'no fraying', 'resists fraying', 'tip hasn\'t worn down', 'holds up to heavy use','no splitting', "doesn't split", 'tip is still intact', 'no signs of wear'],
            '负面-磨损/分叉': ['fray', 'fraying', 'frayed tip', 'frays easily', 'split', 'splitting', 'split nib','wears out quickly', 'wear down fast', 'tip wear', 'tip is gone'],
            '正面-保形性好/硬度佳':['retains its shape', 'holds its point', 'keeps its point', 'point stays sharp', 'tip is still sharp',"doesn't get mushy", "doesn't go flat", 'springs back nicely', 'good snap'],
            '负面-形变/软化': ['gets mushy', 'too soft', 'tip softened', 'spongy tip', 'loses its point', 'lost its fine point', 'point went dull', 'no longer sharp', 'deformed', 'lose its shape', 'went flat', 'lost its snap', "doesn't spring back"],
            '正面-坚固/抗损坏':['durable tip', 'sturdy tip', 'robust nib', 'heavy duty', 'resilient tip', 'holds up to pressure','doesn\'t bend', 'doesn\'t break', 'unbreakable tip', 'withstands abuse'],
            '负面-意外损坏': ['bent tip', 'breaks easily', 'snapped', 'snapped off', 'cracked tip', 'chipped tip', 'broke', 'broken', 'damaged tip', 'tip fell out', 'pushed the tip in', 'tip receded'],
            '负面-寿命不匹配': ['tip wore out before ink ran out', 'felt tip died before the ink','plenty of ink left but tip is useless', 'tip dried out but pen is full', 'nib is gone but still has ink'],
            '正面-寿命长': ['long lasting tip', 'tip lasts a long time', 'tip outlasts the ink', 'good longevity'],
            },
        '流畅性': {
            '正面-书写流畅': ['writes smoothly', 'buttery smooth', 'glides on surface', 'effortless application', 'flows like a dream', 'like silk', 'no resistance', 'smooth as butter', 
                         'glides across the canvas', 'seamless flow', 'very fluid'],
            '负面-干涩/刮纸/断墨': ['scratchy', 'feels scratchy on paper', 'scratches the paper', 'scratchy nib','writes dry', 'arrived dried out', 'dried up quickly', 'pen is dry', 'ink seems dry',
                         'skips', 'skipping', 'skips constantly', 'ink skips', 'hard start', 'hard starts', 'stops and starts','inconsistent flow', 'uneven ink flow', 'ink flow is not consistent',
                        'stops writing', 'stopped working', 'died after a few uses'],
            '负面-出墨过多/漏墨': [ 'blotchy', 'splotchy', 'leaves ink blots', 'ink blobs','too much ink', 'puts down too much ink', 'gushes ink', 'ink gushes out', 'too wet',
                        'feathers badly', 'bleeds everywhere','leaks', 'leaking ink', 'leaked all over', 'ink leaked everywhere', 'leaky pen', 'arrived leaking'],
            '正面-速干且平整': [ 'dries flat', 'no streaks', 'smooth finish', 'not watery', 'solid line', 'vibrant coverage', 'thick paint feel', 'professional finish', 
                        'uniform texture', 'velvety finish', 'opaque delivery'],
            '负面-漆感差/稀薄/起泡': [ 'watery ink', 'too thin', 'pigment and water separated', 'faint color flow', 'bubbles in paint', 'air bubbles', 'frothy ink', 'see-through lines', 
                        'weak pigment', 'diluted color', 'runny paint'],
        },
        '墨水特性': {
            '正面-干燥快/防涂抹': ['quick dry', 'dry so fast','fast dry','not smear','not bleed','no bleed', 'not smear or bleed','dries quickly', 'dries instantly', 'dries immediately', 'fast-drying ink','no smear', 'no smudge', 'zero smear', 'zero smudge', 'smear proof', 'smudge proof',
                        'smudge resistant', 'smear resistant', 'doesn\'t smear', 'doesn\'t smudge','good for lefties', 'perfect for left-handed', 'lefty friendly','can highlight over it', 'highlight without smearing'],
            '负面-干燥慢/易涂抹': ['smears easily', 'smudges easily', 'smears across the page', 'smudges when touched', 'takes forever to dry', 'long drying time', 'never fully dries', 'still wet after minutes', 'slow to dry',
                        'not for left-handed', 'not for lefties', 'smears for left-handers', 'gets ink on my hand','smears with highlighter', 'smudges when layering', 'ruined my work by smudging'],
            '正面-环保/安全/无味': ['non-toxic', 'AP certified non-toxic', 'certified non-toxic', 'no harmful chemicals', 'acid-free', 'archival quality', 'archival ink', 'photo safe','safe for kids', 'kid-safe', 'child-safe', 'great for children',
                        'no smell', 'no odor', 'odorless', 'scent-free', 'low odor', 'no fumes', 'no harsh smell', 'doesn\'t smell bad', 'xylene-free'],
            '负面-气味难闻': ['bad smell', 'strong smell', 'chemical smell', 'toxic smell', 'horrible odor', 'awful scent','overpowering smell', 'overwhelming fumes', 'nauseating smell', 'smells terrible',
                     'stinks', 'reek', 'stench', 'acrid smell', 'plastic smell','gives me a headache', 'headache inducing', 'smell is too strong', 'lingering smell'],
            '正面-持久/防水': ['truly permanent', 'permanent bond', 'archival quality', 'archival ink', 'museum quality','is waterproof', 'water resistant', 'doesn\'t run with water', 'survives spills', 'water-fast',
                      'fade proof', 'fade resistant', 'lightfast', 'excellent lightfastness', 'uv resistant', 'doesn\'t fade over time'],
            '负面-易褪色/不防水': ['not permanent', 'isn\'t permanent', 'fades quickly', 'fades over time', 'colors have faded', 'not lightfast','not waterproof', 'isn\'t water resistant', 'washes away', 'runs with water', 'smears with water',
                        'ruined by a drop of water', 'ink bleeds when wet'],
            '正面-续航长': ['lasts a long time', 'lasted for months', 'seems to last forever', 'plenty of ink', 'large ink capacity', 'still going strong', 'has a lot of ink', 'haven\'t run out yet',
                    'great longevity', 'long-lasting ink'],
            '负面-消耗快': ['runs out quickly', 'ran out of ink fast', 'dries out too fast', 'died quickly','empty fast', 'used up too fast', 'ran dry very quickly', 'doesn\'t last long','run out of paint',
                     'ink runs out in a day', 'not much ink inside', 'low ink capacity', 'wish it held more ink'],
            '正面-金属效果好': ['great metallic effect', 'nice metallic sheen', 'shiny metal finish', 'strong metallic look', 'looks like real metal', 'beautiful chrome finish', 'very reflective'],
            '负面-金属效果差': ['dull metallic', 'not shiny', 'no metallic effect', 'looks flat', 'weak sheen', 'not reflective'],
            '正面-闪光效果好': ['lots of glitter', 'beautiful shimmer', 'sparkly', 'glitter is vibrant', 'nice pearlescent effect', 'very glittery', 'good sparkle'],
            '负面-闪光效果差': ['not enough glitter', 'no shimmer', 'glitter falls off', 'dull sparkle', 'barely any glitter', 'messy glitter'],
            '正面-荧光/霓虹效果好': ['neon pops', 'very bright neon', 'glows under blacklight', 'super fluorescent', 'vibrant neon', 'glows nicely'],
            '负面-荧光/霓虹效果淡': ['neon is dull', 'not very bright', "doesn't glow", 'not a true neon color', 'disappointing neon'],
            '负面-荧光/霓虹效果过饱和': ['too neon', 'too bright', 'too fluorescent', 'too neon/bright'],
            '正面-变色效果好': ['love the color change', 'chameleon effect is stunning', 'shifts colors beautifully', 'works in the sun', 'heat sensitive works'],
            '负面-变色效果差': ["doesn't change color", 'color shift is weak', 'barely changes', 'no chameleon effect'],
            '正面-夜光效果好': ['glows brightly in the dark', 'long lasting glow', 'charges quickly', 'very luminous'],
            '负面-夜光效果差': ["doesn't glow", 'glow is weak', 'fades too fast', 'barely glows'],
            '正面-香味好闻': ['smells great', 'love the scent', 'nice fragrance', 'fun scents', 'smells like fruit'],
            '负面-香味难闻/太浓': ['smell is too strong', 'bad smell', "doesn't smell like anything", 'chemical smell', 'artificial scent'],
            '中性-提及干燥/涂抹': ['drying time', 'dry time', 'smudge proof', 'smear proof', 'for left-handed', 'for lefties'],
            '中性-提及气味/安全': ['odor', 'smell', 'fumes', 'scent', 'non-toxic', 'acid-free', 'archival quality', 'chemical smell', 'safe for kids'],
            '中性-提及持久/防水': ['waterproof', 'water resistance', 'fade proof', 'lightfastness', 'lightfast rating', 'permanent ink', 'archival ink'],
            '中性-提及续航': ['longevity', 'ink life', 'how long they last', 'runs out quickly', 'runs dry', 'ink capacity'],
            '中性-提及金属效果': ['metallic ink', 'metallic effect', 'metallic sheen', 'chrome finish', 'reflective properties'],
            '中性-提及闪光效果': ['glitter ink', 'shimmer effect', 'sparkle', 'pearlescent effect', 'glitter particles'],
            '中性-提及荧光/霓虹效果': ['neon ink', 'fluorescent colors', 'under blacklight', 'glowing effect'],
            '中性-提及变色效果': ['color changing', 'color shift', 'chameleon effect', 'heat sensitive'],
            '中性-提及夜光效果': ['glow in the dark', 'luminous ink', 'glowing properties'],
            '中性-提及香味': ['scented ink', 'scented markers', 'fruit scent', 'fragrance'],
            '中性-提及可擦除性': ['erasable ink', 'can be erased', 'erases cleanly', 'frixion ink'],
        },
        '笔身与易用性': {
            '正面-材质/做工好': ['durable body', 'sturdy', 'sturdy build', 'well-made', 'solid construction', 'solidly built','quality feel', 'feels premium', 'high quality materials', 'quality build', 'well put together','feels substantial', 'built to last', 'high-grade plastic', 'metal construction', 'feels expensive'],
            '负面-材质/做工差': ['feels cheap', 'flimsy', 'cheap plastic', 'thin plastic', 'brittle plastic', 'feels plasticky', 'poorly made', 'poor construction', 'badly made', 'low quality build', 'fell apart',
                       'cracked easily', 'developed a crack','break', 'broke easily', 'broke when dropped', 'snapped in half', 'easy to break','doesn\'t feel durable', 'not sturdy'],
            '正面-握持舒适': ['comfortable to hold', 'comfortable grip', 'ergonomic', 'ergonomic design', 'ergonomic shape', 'nice to hold', 'feels good in the hand', 'feels great in the hand', 'good grip', 'soft grip',
                      'well-balanced', 'perfect weight', 'nice balance', 'fits my hand perfectly', 'contours to my hand', 'doesn\'t cause fatigue', 'no hand cramps', 'can write for hours', 'can draw for hours', 'reduces hand strain'],
            '负面-握持不适': [ 'uncomfortable to hold', 'uncomfortable grip', 'awkward to hold', 'awkward shape','causes hand fatigue', 'tires my hand quickly', 'gives me hand cramps', 'hand cramps up', 'hurts my hand', 'digs into my hand', 'sharp edges', 'too thick', 'too thin', 'too wide', 'too narrow', 'slippery grip', 'hard to get a good grip', 'poorly balanced', 'too heavy', 'too light', 'weird balance'],
            '正面-笔帽体验好': ['cap posts well', 'posts securely', 'cap posts nicely', 'secure fit', 'cap fits snugly', 'airtight seal', 'seals well', 'tight seal', 'cap clicks shut', 'satisfying click', 'audible click', 'easy to open cap', 'easy to uncap', 'cap stays on', 'doesn\'t dry out', 'durable clip'],
            '负面-笔帽体验差': ['hard to open cap', 'cap is too tight', 'difficult to uncap', 'struggle to open','loose cap', 'cap falls off', "cap doesn't stay on", "doesn't seal properly", 'not airtight',
                       'pen dried out because of the cap', 'dries out quickly','cracked cap', 'cap broke', 'cap broke easily', 'brittle cap','cap won\'t post', 'doesn\'t post securely', 'cap is too loose to post','clip broke off', 'flimsy clip', 'weak clip'],
            '正面-易于使用': ['easy to use', 'simple to use', 'user-friendly', 'intuitive design', 'no learning curve', 'effortless to use', 'easy to handle', 'easy to control', 'good control',],
            '中性-提及材质/做工': ['pen body', 'body material', 'barrel material', 'build quality', 'construction', 'plastic body', 'metal body', 'wooden body', 'resin body', 'surface finish', 'pen finish'],
            '中性-提及握持': ['grip section', 'grip comfort', 'ergonomic grip', 'pen balance', 'pen shape', 'barrel diameter',  'how it feels in the hand'],
            '中性-提及笔帽': ['pen cap', 'pocket clip', 'posting the cap', 'cap seal', 'screw cap', 'snap cap','caps','cap'],
            '中性-提及易用性/便携': ['portability', 'easy to carry', 'travel case', 'pen roll', 'for on the go', 'pocket pen', 'travel friendly'],
        },
        '绘画表现': {
            '正面-线条表现好/可控': ['good control', 'controllable lines', 'great line variation', 'crisp lines', 'consistent lines', 'clean lines', 'no skipping', 'sharp lines', 'great for fine details'],
            '负面-线条表现差/难控': ['hard to control', 'inconsistent line', 'uncontrollable', 'not for details', 'wobbly lines', 'shaky lines', 'broken line'],
            '正面-覆盖力好/不透明': ['opaque', 'good coverage', 'covers well', 'one coat', 'hides underlying color', 'works on dark paper', 'great opacity'],
            '负面-过于透明/覆盖力差': ['not opaque', 'too sheer', "doesn't cover", 'needs multiple coats', 'see through'],
            '正面-涂色均匀': ['even application', 'smooth application', 'no streaks', 'self-leveling', 'consistent color', 'no streaking'],
            '负面-涂色不均': ['streak', 'streaky', 'streaking', 'leaves streaks', 'patchy', 'blotchy'],
            '正面-兼容铅笔': ['goes over pencil cleanly', "doesn't smudge graphite", 'erases pencil underneath', 'covers pencil lines well'],
            '负面-铅笔兼容性差': ['smears pencil lines', 'smudges graphite', 'lifts graphite', 'muddy with pencil', "doesn't cover pencil"],
            '正面-兼容勾线笔': ["doesn't smear fineliner", 'works with micron pens', 'layers over ink', 'copic-proof ink compatible', 'safe over ink'],
            '负面-勾线笔兼容性差': ['smears fineliner ink', 'reactivates ink', 'lifts the ink line', 'bleeding with ink lines', 'makes ink run'],
            '正面-兼容水彩/水粉': ['layers over watercolor', 'works well with gouache', 'can use for watercolor effects', "doesn't lift watercolor"],
            '负面-水彩/水粉兼容性差': ['lifts watercolor', 'muddy with gouache', 'reactivates paint underneath', 'smears watercolor'],
            '正面-兼容彩铅': ['layers well with colored pencils', 'good for marker and pencil', 'blends with pencil crayon', 'works over wax pencil'],
            '负面-彩铅兼容性差': ['waxy buildup with colored pencils', "doesn't layer over pencil crayon", 'reacts weirdly with other markers'],
            '正面-兼容丙烯马克笔': ['layers nicely over Posca', 'can draw on top of Posca', "doesn't lift the acrylic", 'good with acrylic markers', 'adheres well to paint'],
            '负面-不兼容丙烯马克笔': ['smears Posca paint', "doesn't stick to acrylic marker", 'lifts the underlying acrylic', 'scratches off the acrylic surface'],
            '中性-提及线条表现': ['line quality', 'line control', 'line variation', 'stroke consistency', 'stroke', 'fine details', 'detailed work'],
            '中性-提及覆盖力': ['opacity', 'coverage', 'sheer', 'transparency', 'opaque', 'single coat', 'coverage strength'],
            '中性-提及涂色均匀性': ['even application', 'smooth application', 'streaks', 'streaky', 'patchy', 'blotchy', 'self-leveling'],
            '中性-提及可再加墨': ['reactivate', 'reactivation', 'lift', 'lifting', 'movable ink', 're-wettable'],
            '中性-提及兼容铅笔': ['over pencil', 'with pencil',],
            '中性-提及兼容勾线笔': ['over ink', 'with ink', 'over fineliner', 'with fineliner', 'over micron', 'copic-proof'],
            '中性-提及兼容水彩/水粉': ['over watercolor', 'with watercolor', 'with gouache', 'on top of paint'],
            '中性-提及兼容彩铅': ['over colored pencils', 'with colored pencils', 'over pencil crayon', 'with wax pencil'],
            '中性-提及兼容丙烯马克笔': ['on top of acrylic', 'over acrylic', 'with acrylic markers', 'with posca', 'on paint marker'],
        },
        '场景表现': {
            '正面-适合大面积填色': ['great for coloring', 'good for large areas', 'fills spaces evenly', 'no streaking in large blocks', 'coloring book friendly', 'smooth coverage'],
            '负面-不适合大面积填色': ['streaky when coloring', 'dries too fast for large areas', 'bad for filling large spaces', 'leaves marker lines', 'patchy on large areas'],
            '正面-适合漫画/动漫创作': ['great for manga', 'perfect for comics', 'blends skin tones beautifully', 'works for anime style', 'good for cel shading', 'great for character art'],
            '负面-不适合漫画/动漫创作': ['hard to blend skin tones', "colors aren't right for manga", 'smears my line art', 'not good for comic art'],
            '正面-适合插画创作': ['great for illustration', 'professional illustration results', 'layers beautifully for art', 'vibrant illustrations', 'perfect for artists'],
            '负面-不适合插画创作': ['not for professional illustration', 'colors are not vibrant enough for art', 'muddy blends for illustration', 'hobby grade only'],
            '正面-适合着色书/填色': ['great for coloring books', 'perfect for adult coloring', 'coloring book friendly', 'no bleed in coloring book', "doesn't ghost on coloring pages", 'safe for single-sided books', 'fine tip is perfect for intricate designs', 'great for mandalas', 'gets into tiny spaces'],
            '负面-不适合着色书/填色': ['not for coloring books', 'ruined my coloring book', 'bleeds through every page', 'ghosting is too bad for coloring books', 'ruined the next page', 'tip is too broad for detailed coloring', 'bleeds outside the lines in small patterns', 'pills the coloring book paper', 'tears the paper'],
            '正面-适合书法/手写艺术': ['perfect for calligraphy', 'great for hand lettering', 'nice thick and thin strokes', 'good for upstrokes and downstrokes', 'flexible tip for lettering', 'rich black for calligraphy'],
            '负面-不适合书法/手写艺术': ['tip is too stiff for calligraphy', 'hard to control line variation', 'ink feathers during lettering', 'not good for brush lettering', 'ink is not dark enough for calligraphy'],
            '正面-适合手工艺/物品定制': ['great for diy projects', 'perfect for customizing shoes', 'works on canvas bags', 'permanent on rocks and wood', 'good for crafting'],
            '负面-不适合手工艺/物品定制': ['wipes off from plastic', 'not for outdoor use', 'color fades on fabric', "doesn't work on sealed surfaces"],
            '正面-适合儿童/教学': ['great for kids', 'safe for children', 'non-toxic', 'washable ink', 'durable tip for heavy hands', 'bright colors for kids', 'good for classroom use'],
            '负面-不适合儿童/教学': ['strong smell not for kids', 'ink stains clothes', 'tip broke easily with pressure', 'cap is hard for a child to open'],
            '正面-适合刻字/细节': ['perfect for lettering', 'great for calligraphy', 'nice for writing greetings', 'fine tip for small details', 'beautiful for sentiments'],
            '负面-不适合刻字/细节': ['too thick for lettering', 'bleeds when writing', 'hard to do calligraphy with'],
            '正面-多表面DIY': ['perfect for rock painting', 'works great on wood', 'customizing sneakers', 'painting on glass', 'ceramic decorating', 'canvas art', 'outdoor decor', 'painting pumpkins', 'ornament decorating'],
            '负面-不能多表面DIY': ['scrapes off glass', 'not permanent on plastic', 'faded on outdoor rocks', 'ink doesn't stick to metal'],
            '中性-提及大面积填色': ['coloring large areas', 'filling in spaces', 'large coverage', 'background coloring'],
            '中性-提及漫画/动漫创作': ['manga', 'comic art', 'anime art', 'line art', 'character art', 'cel shading'],
            '中性-提及插画创作': ['illustration', 'illustrating', 'artwork', 'for my illustrations'],
            '中性-提及着色书/填色': ['coloring book', 'coloring books', 'adult coloring', 'colouring book', 'mandala', 'mandalas', 'intricate designs', 'coloring pages', 'secret garden', 'johanna basford', 'color by number'],
            '中性-提及书法/手写艺术': ['calligraphy', 'hand lettering', 'lettering practice', 'upstrokes', 'downstrokes','typography'],
            '中性-提及手工艺/物品定制': ['diy project', 'craft project', 'crafting with', 'customizing shoes', 'on canvas bags', 'on rocks', 'on wood', 'on plastic', 'on sealed surfaces'],
            '中性-提及儿童/教学': [ 'for kids', 'for children', 'in the classroom', 'for my students', 'art class', 'school project'],
            '中性-提及刻字/细节': ['lettering for cards', 'writing greetings', 'writing sentiments', 'for small details', 'for fine details', 'detailed work'],

        },
        '表面/介质表现': {
            '正面-在纸张上表现好': ['works great on marker paper', 'smooth on bristol board', 'blends well on bleedproof paper', 'perfect for mixed media paper', 'fit for paper', 'good for paper'],
            '负面-在纸张上表现差': ['still bleeds through marker paper', 'feathers on hot press paper', 'destroys bristol surface', 'pills my cold press paper', 'mess up your paper'],
            '中性-提及纸张': ['on paper','marker paper', 'bristol board', 'bristol', 'watercolor paper', 'mixed media paper', 'bleedproof paper', 'hot press', 'cold press', 'sketch book','for paper','for papers'],
            '正面-在深色纸张上显色好': ['opaque on black paper', 'shows up well on dark paper', 'great coverage on kraft paper', 'vibrant on colored paper', 'pops on black', 'shows up beautifully', 'great on black cardstock'],
            '负面-在深色纸张上显色效果差': ['not opaque on black', 'disappears on dark paper', 'too transparent for colored paper', "doesn't show up", 'color looks dull on black'],
            '中性-提及深色纸张': ['black paper', 'dark paper', 'kraft paper', 'colored paper'],
            '正面-在布料上效果好': ['great on fabric', 'permanent on t-shirt', 'holds up in the wash', 'vibrant on textile', 'perfect for customizing shoes', "doesn't feather on cotton", 'survived the wash', 'applies smoothly to canvas', 'flexible on fabric', 'heat sets perfectly', "doesn't stiffen the fabric"],
            '负面-在布料上效果差': ['bleeds on fabric', 'feathers on canvas', 'fades after washing', 'washes out', 'makes the fabric stiff', 'washed right out', 'faded after one wash', 'cracked on the fabric', 'cracks when fabric flexes'],
            '中性-提及布料': ['canvas','canvas mural','on fabric', 'on canvas', 'on t-shirt', 'on textile', 'on cotton', 'on denim', 'for fabric', 'fabric marker'],
            '正面-在木材上表现好': ['great on wood', 'vibrant color on wood', 'dries nicely on wood', 'perfect for wood crafts', 'sharp lines on wood', 'beautiful finish on wood', 'seals nicely', 'vibrant on unfinished wood'],
            '负面-在木材上表现差': ['bleeds into the wood grain', 'color looks dull on wood', 'uneven color on wood', 'smears on sealed wood', 'bleeds with the grain', 'raised the wood grain', 'makes the grain swell'],
            '中性-提及木材': ['on wood', 'for wood', 'writes on wood', 'draw on wood', 'wood grain', 'sealed wood', 'wood crafts', 'unfinished wood'],
            '正面-在石头上表现好': ['great for rock painting', 'vibrant on rocks', 'opaque on stone', 'smooth lines on rocks', 'durable on pebbles', 'covers rocks smoothly', 'perfect for rock art', 'adheres well to stone', 'weather resistant', 'dries quickly on rocks'],
            '负面-在石头上表现差': ['scratches off rocks', 'not opaque enough for stone', 'color is dull on rocks', 'clogs tip on rough stone', 'hard to draw on rocks', 'chips off easily', 'too watery for rocks', 'streaky'],
            '中性-提及石头': ['on rock', 'on rocks', 'on stone', 'on stones', 'on pebble', 'on pebbles', 'for rocks', 'for rock painting', 'rock painting'],
            '正面-在粘土上表现好': ['works on polymer clay', 'great on air dry clay', 'vibrant on clay', 'soaks in nicely on bisque', "doesn't react with sealant", 'adheres perfectly to clay', 'bakes well', 'color stays true after sealing'],
            '负面-在粘土上表现差': ["doesn't adhere to clay", 'smears on polymer clay', 'clogs tip on un-sanded clay', 'reactivates the clay', 'melts the clay surface', 'never fully cures on clay', 'smears easily on polymer clay', 'reacts with glaze'],
            '中性-提及粘土': ['on clay', 'on polymer clay', 'on air dry clay', 'on bisque', 'for clay'],
            '正面-在玻璃(Glass)上表现好': ['permanent on glass', 'smudge proof on glass', 'crisp lines on glass', 'adheres well to glass', 'opaque on glass', 'vibrant on glass', 'writes smoothly on glass', 'removable with windex'],
            '负面-在玻璃(Glass)上表现差': ['wipes off glass', 'smears on glass', 'scratches off glass', 'beads up on glass', 'streaky on glass', 'difficult to remove from glass'],
            '中性-提及玻璃(Glass)': ['on glass', 'for glass', 'writes on glass', 'glass art', 'stain glass'],
            '正面-在陶瓷(Ceramic)上表现好': ['permanent on ceramic', 'writes on mugs', 'decorating ceramic', 'dishwasher safe', 'vibrant on ceramic', 'bake to set', 'cures to a hard finish', 'perfect for customizing mugs', 'great on mugs'],
            '负面-在陶瓷(Ceramic)上表现差': ['never dries on ceramic', 'wipes off ceramic', 'smears on ceramic', 'not dishwasher safe', 'washes off mug', 'scratches off ceramic', 'fades after baking', 'comes right off in dishwasher'],
            '中性-提及陶瓷(Ceramic)': ['on ceramic', 'on mugs', 'on glazed surface', 'for ceramic', 'decorating ceramic'],
            '正面-在塑料(Plastic)上表现好': ['permanent on plastic', 'smudge proof on plastic', 'adheres to plastic', 'vibrant on plastic', 'bonds to plastic', 'dries instantly on plastic', 'great on plastic models'],
            '负面-在塑料(Plastic)上表现差': ['wipes off plastic', 'smears on plastic', "doesn't stick to plastic", 'never dries on plastic', 'rubs off plastic', 'eats the plastic', 'remains sticky on plastic', 'remains tacky'],
            '中性-提及塑料(Plastic)': ['on plastic', 'for plastic', 'writes on plastic', 'plastic models'],
            '正面-在金属(Metal)上表现好': ['adheres to metal', 'permanent on metal', "doesn't scratch off metal", 'clean lines on metal', 'opaque on metal', 'dries quickly on metal', 'marks metal clearly', 'great for metalwork', 'weather resistant on metal'],
            '负面-在金属(Metal)上表现差': ['scratches off metal', 'smears on metal', 'wipes off metal', 'flaked off', 'peeled off metal', 'corrodes metal', 'takes forever to dry on metal', 'rubs off easily', "doesn't adhere to aluminum"],
            '中性-提及金属(Metal)': ['on metal', 'on aluminum', 'for metal', 'marks on metal'],
            '正面-在墙面上表现好': ['great coverage on walls', 'opaque on painted surfaces', 'covers in one coat', 'permanent on drywall', 'durable for murals', 'weatherproof', 'smooth on walls', 'great for mural work', 'low-fume for indoor use'],
            '负面-在墙面上表现差': ['wipes off the wall', 'not for outdoor murals', 'too transparent for walls', 'streaky on walls', 'damaged my wall'],
            '中性-提及墙面': ['on the wall', 'on walls', 'for murals', 'graffiti', 'on drywall', 'on plaster', 'on painted wall'],
        },
        '外观与包装': {
            '正面-外观/设计美观': ['beautiful design', 'minimalist design', 'sleek design', 'clean design', 'well-designed','thoughtful design', 'love the design', 'love the look of', 'pleasing aesthetic', 'looks elegant', 'high-end look', 'modern look', 'looks professional', 'impressed with the design'],
            '负面-外观廉价/丑': ['looks cheap', 'feels cheap', 'cheaply made', 'cheap appearance', 'low-end look', 'plasticky feel', 'flimsy appearance', 'looks like a toy', 'toy-like', 'looks like a child\'s toy','ugly design', 'unattractive design', 'clunky design', 'awkward look', 'poorly designed', 'gaudy colors', 'tacky design', 'looks dated', 'outdated design'],
            '正面-包装美观/保护好': ['beautiful packaging', 'nice packaging', 'lovely box', 'great presentation', 'well presented', 'elegant packaging', 'giftable', 'perfect for a gift', 'great gift box', 'nice enough to gift','well packaged', 'packaged securely', 'protective packaging', 'arrived safe', 'arrived in perfect condition', 'no damage during shipping', 'excellent packaging',
                        'sturdy case', 'durable case', 'high-quality box', 'nice tin', 'reusable case', 'great storage tin', 'comes in a nice case'],
            '负面-包装廉价/易损坏': ['flimsy packaging', 'cheap packaging', 'thin cardboard', 'poor quality box', 'doesn\'t protect the pens','damaged box', 'crushed box', 'dented tin', 'arrived damaged', 'damaged in transit', 'damaged during shipping','broken case', 'cracked case', 'case was broken', 'clasp broke', 'latch doesn\'t work', 'zipper broke','cheap case', 'flimsy case', 'case arrived broken'],
            '正面-收纳便利': ['well-organized', 'keeps them neat', 'keeps them organized', 'easy to organize', 'easy access to colors', 'easy to find the color', 'easy to get pens out', 'convenient storage', 'handy case', 'sturdy case', 'nice carrying case', 'protective case', 'pens fit perfectly', 'individual slots for each pen', 'great storage box', 'useful pen holder'],
            '负面-收纳不便': ['hard to get out', 'difficult to remove pens', 'pens are too tight in the slots', 'struggle to get them out','messy organization', 'poorly organized', 'pens fall out of place', 'don\'t stay in their slots', 'no individual slots', 'pens are all jumbled together', 'hard to put pens back','case doesn\'t close', 'case doesn\'t latch', 'lid won\'t stay closed', 'clasp broke', 'zipper broke','flimsy trays', 'pens fall out when opened'],
            '中性-提及外观': ['pen design', 'overall look', 'visual appeal', 'aesthetic', 'appearance', 'form factor', 'finish', 'color scheme'],
            '中性-提及包装': ['packaging', 'box', 'outer box', 'sleeve', 'tin case', 'gift box', 'presentation', 'protective case', 'blister pack', 'unboxing'],
            '中性-提及收纳': ['storage case', 'carrying case', 'pen holder', 'pen stand', 'pen roll', 'organizer tray', 'layout of the tray', 'how they are organized'],
        },
        '多样性与适配性': {
            '正面-用途广泛': ['multi-purpose', 'all-in-one', 'jack of all trades', 'works for everything','use it for everything', 'handles a variety of tasks', 'works on multiple surfaces',
                      'use on different surfaces', 'good for many different projects', 'one set for all my needs','great for both drawing and writing'],
            '负面-用途单一': ['not versatile', 'lacks versatility', 'not multi-purpose', 'single-purpose', 'single use','one-trick pony', 'limited use', 'very limited in its use', 'limited application',  'only for paper', 'only works on paper', 'doesn\'t work on other surfaces',
                     'only good for one thing', 'useless for anything else', 'very specific use'],
            '正面-可拓展性 (Collection can be expanded)': [ 'expandable collection', 'can add to my collection', 'love adding to my collection', 'complete my collection', 'collect all the colors', 'love that they release new sets', 'new colors available',
                                      'hope they release more colors', 'can\'t wait for new colors','limited edition colors', 'love the special editions', 'collector\'s edition'],
            '负面-可拓展性差 (Poor expandability)': ['no new colors', 'collection is limited', 'wish they had more shades', 'no new sets released','stagnant collection', 'line seems to be discontinued', 'never release new colors',
                                   'can\'t expand my collection', 'no updates to the color range', 'stuck with the same colors', 'wish they would expand the range', 'color range is too small', 'no new releases'],
            '正面-可补充性 (Can be replenished)': ['refillable', 'refillable ink', 'ink refills available', 'can buy refills', 'replaceable cartridges','buy individually', 'can buy single pens', 'sold individually', 'available as singles', 'open stock',
                                 'don\'t have to buy the whole set', 'can just replace the one I need','replaceable nibs', 'can replace the nibs', 'replacement nibs available'],
            '负面-可补充性差 (Poor replenishability)': ["can't buy single", 'not sold individually', 'not available individually', 'can\'t buy individual pens', 'not sold as singles','wish they sold refills', 'no refills available', 'can\'t find refills', 'ink is not refillable', 'no refill cartridges', 'no replacement nibs', 'can\'t replace the tip', 'no replacement parts',
                                   'have to buy a whole new set', 'forced to rebuy the set', 'must buy the entire set again'],
            '正面-单支购买': ['can buy single white pens', 'available as individual markers', 'don\'t need to buy a whole pack for one color', 'sold as singles for replacement'],
            '负面-不可单支购买': ['wasteful to buy a new set', 'no single replacements', 'can\'t find individual pens for sale', 'forced to buy 12 just for the black one'],
            '中性-提及用途广泛性': ['versatility', 'multi-purpose', 'all-in-one', 'works on multiple surfaces', 'use for different things', 'all purpose', 'various uses'],
            '中性-提及可拓展性': ['expandable collection', 'add to the collection', 'complete the set', 'new colors','new sets released', 'limited edition', 'collect all the colors'],
            '中性-提及可补充性': ['refillable', 'open stock', 'sold individually', 'buy single pens', 'replacement nibs', 'ink refills', 'refill cartridges'],
            
            },
        '教育与启发': {
            '正面-激发创意/乐趣': ['fun to use', 'so much fun to play with', 'a joy to use', 'enjoyable to use', 'very satisfying','inspires me to create', 'makes me want to draw', 'makes me want to create', 'sparks my creativity',
                        'boosts my creativity', 'unleashes creativity', 'creative juices are flowing','gets me out of a creative block', 'helps with creative block', 'opens up new possibilities'],
            '正面-适合初学者': ['beginner friendly', 'good for beginners', 'easy for a beginner', 'perfect for beginners','easy to start', 'great starting point', 'just starting out', 'getting started', 'starter kit', 'great starter set', 'my first set', 'new to art', 'new to painting', 'new to drawing', 'first time trying','easy to learn', 'easy to learn with', 'no learning curve', 'no prior experience needed'],
            '负面-有学习门槛': ['steep learning curve', 'learning curve', 'not for beginners', 'not beginner friendly','hard to use', 'difficult to use', 'confusing to use', 'not intuitive', 'hard to control',
                       'difficult to get the hang of', 'takes a lot of practice', 'requires a lot of skill','frustrating for a beginner', 'not easy to get started with'],
            '正面-有教学支持': ['helpful guide', 'clear instructions', 'easy to follow guide', 'step-by-step guide',  'well-written instructions', 'great instruction book','good tutorial', 'helpful video tutorial', 'easy to follow tutorial','great community', 'supportive community', 'helpful facebook group', 'comes with practice sheets', 'love the worksheets', 'great online course'],
            '负面-无教学支持': ['no instructions', 'no guide included', 'didn\'t come with instructions', 'no user manual', 'lacks instructions', 'confusing guide', 'unhelpful guide', 'hard to understand instructions', 'instructions are not clear', 'useless instructions', 'poorly written', 'vague instructions', 'bad translation','instructions in another language', 'only in chinese',
                       'no online tutorials', 'can\'t find any videos on how to use'],
            '中性-提及创意/乐趣': [ 'creative juices', 'fun activity', 'joy of creating', 'spark creativity', 'boost creativity','creative outlet', 'artistic expression', 'fun to use', 'enjoyable process', 'doodling for fun'],
            '中性-提及学习门槛': ['beginner friendly', 'good for beginners', 'easy for a beginner','starter kit', 'starter set', 'my first set', 'entry-level','learning curve', 'no prior experience', 'easy to learn with',
                        'just starting out', 'getting started','new to art', 'new to painting', 'new to drawing','learning to draw', 'learning to paint'],
            '中性-提及教学支持': ['instruction book', 'instructional booklet', 'guidebook', 'step-by-step guide', 'how-to guide', 'learning guide', 'video tutorial', 'youtube tutorial', 'following a tutorial',
                        'online course', 'skillshare class', 'practice sheets', 'worksheets','online community', 'facebook group'],
        },
        '特殊用途': {
            '正面-专业级表现': ['professional grade', 'artist grade', 'pro grade', 'professional quality', 'artist quality', 'studio grade', 'museum quality', 'for serious artists', 'not student grade','professional results', 'gallery quality results', 'publication quality',
                       'industry standard', 'lightfast', 'excellent lightfastness', 'high lightfastness rating', 'fade-resistant', 'fade proof', 'archival quality', 'archival ink', 'archival pigment'],
            '负面-非专业级': ['not professional grade', 'not artist grade', 'hobby grade', 'student grade', 'for hobby use only', 'for casual use only', 'not for serious artists',
                      'feels like a toy', 'not for client work', 'not for commissions','not archival', 'not lightfast', 'more of a toy than a tool'],
            '中性-提及专业性': ['professional grade', 'artist grade', 'hobby grade', 'student grade', 'pro grade', 'lightfast', 'lightfastness rating', 'archival quality', 'archival ink', 'museum quality'],
        },
        '性价比': {
            '正面-性价比高': ['affordable', 'cheap', 'good value', 'great deal', 'worth the money', 'great buy', 'reasonable price', 'cheaper than', 'alternative to','excellent value', 'amazing value','inexpensive','low price', 'great price point','money well spent', 'can\'t beat the price'],
            '负面-价格昂贵': ['expensive', 'overpriced', 'not worth', 'pricey', 'costly', 'rip off', 'too much', 'waste of money','not worth it','over-priced'],

            },
        '配套与服务(色卡)': {
            '正面-提供色卡/好用': ['comes with a swatch card', 'includes a swatch card', 'love the swatch card', 'helpful swatch card', 'great for swatching', 'easy to swatch', 'blank swatch card', 'pre-printed swatch card'],
            '负面-缺少色卡/不好用': ['no swatch card', "wish it had a swatch card", "doesn't come with a swatch card", 'had to make my own swatch card', 'swatch card is inaccurate', 'swatch card is useless', "colors on swatch card don't match"],
            '中性-提及色卡': ['swatch card',  'color chart'],
            },
        '购买与服务体验': {
            '正面-开箱/展示': ['beautiful presentation', 'great unboxing experience', 'perfect for a gift', 'looks professional', 'elegant packaging', 'giftable', 'nice gift box', 'well presented', 'impressive presentation',
                       'lovely box', 'makes a great gift', 'nicely laid out'],
            '负面-运输/损坏': ['arrived broken', 'pens arrived broken', 'some were broken', 'cracked on arrival', 'damaged during shipping','damaged in transit', 'arrived damaged', 'item was damaged','leaking ink', 'leaked all over', 'ink leaked everywhere', 'arrived leaking','box was crushed',
                      'package was damaged', 'box was open', 'dented tin', 'poorly packaged for shipping', 'not well protected', 'arrived in bad shape'],
            '正面-客服/售后': ['great customer service', 'excellent customer service', 'amazing support', 'seller was helpful', 'seller was very helpful', 'very responsive seller', 'quick response', 'fast reply',
                      'answered my questions quickly', 'resolved my issue quickly', 'problem solved','fast replacement', 'quick replacement', 'sent a replacement right away', 'easy replacement process',
                       'easy refund', 'hassle-free refund', 'full refund was issued','went above and beyond', 'proactive customer service'],
            '负面-客服/售后': ['bad customer service', 'terrible customer service', 'poor support', 'no customer service','seller was unresponsive', 'no response from seller', 'never replied', 'took forever to respond', 'slow response',
                      'seller was unhelpful', 'refused to help', 'unwilling to help', 'could not resolve the issue','missing items', 'missing parts', 'didn\'t receive all items',
                       'wrong item sent', 'received the wrong color', 'sent the wrong size','difficult return process', 'hassle to get a refund', 'refused a refund', 'no replacement offered'],
            '中性-提及开箱/展示': ['unboxing experience', 'presentation', 'packaging', 'giftable', 'nice box', 'sturdy case', 'storage tin', 'well organized', 'comes in a case'],
            '中性-提及运输': ['shipping', 'delivery', 'arrival condition', 'transit', 'shipped', 'arrived',  'damage','damaged', 'broken', 'crushed', 'leaking', 'shipping box', 'protective packaging'],
            '中性-提及客服/售后': ['customer service', 'contacted seller', 'contacted support', 'seller response','replacement', 'refund', 'return process', 'exchange', 'missing items', 'wrong item sent', 'issue resolved'],
        }
}


# --- 2. 文本分析引擎 ---
def analyze_features(text_series):
    """
    输入评论序列，返回各维度的 Highlight 和 Pain Point 频次
    """
    results = []
    text_data = text_series.dropna().astype(str).lower()
    
    for category, subtokens in FEATURE_DIC.items():
        # 计算正面词频
        pos_count = 0
        for word in subtokens['Highlights']:
            pos_count += text_data.str.contains(re.escape(word)).sum()
            
        # 计算负面词频
        neg_count = 0
        for word in subtokens['Pain Points']:
            neg_count += text_data.str.contains(re.escape(word)).sum()
            
        results.append({
            "维度": category,
            "Highlight Count": pos_count,
            "Pain Point Count": neg_count,
            "Total Mentions": pos_count + neg_count
        })
    return pd.DataFrame(results)

# --- 3. 页面配置与数据加载 ---
st.set_page_config(page_title="丙烯调研看板", layout="wide")

@st.cache_data
def load_raw_data():
    data_map = {
        "kids_sales.xlsx": ("儿童丙烯", "🔥 高销量 (Top 10)"),
        "kids_trending.xlsx": ("儿童丙烯", "📈 高增长趋势"),
        "large_capacity_sales.xlsx": ("大容量丙烯", "🔥 高销量 (Top 10)"),
        "large_capacity_trending.xlsx": ("大容量丙烯", "📈 高增长趋势")
    }
    combined = []
    for filename, info in data_map.items():
        if os.path.exists(filename):
            try:
                df = pd.read_excel(filename, engine='openpyxl')
                # 假设评论列名为 'Review Body' 或 'content'，请根据实际修改
                review_col = 'Review Body' if 'Review Body' in df.columns else df.columns[0]
                df['review_content'] = df[review_col] 
                df['main_category'] = info[0]
                df['sub_type'] = info[1]
                combined.append(df)
            except Exception as e:
                st.sidebar.error(f"读取 {filename} 失败: {e}")
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

df = load_raw_data()

# --- 4. 侧边栏导航 ---
st.sidebar.header("📂 核心板块选择")
selected_main = st.sidebar.radio("请选择调研产品线：", ["儿童丙烯", "大容量丙烯"])
filtered_df = df[df['main_category'] == selected_main]

# --- 5. 主界面分析布局 ---
if not filtered_df.empty:
    st.title(f"🎨 {selected_main} 深度洞察看板")
    
    # 按照 销量 和 趋势 拆分数据
    sales_data = filtered_df[filtered_df['sub_type'].str.contains("销量")]
    trend_data = filtered_df[filtered_df['sub_type'].str.contains("趋势")]

    # 定义展示函数
    def render_analysis_section(data, title, color):
        st.subheader(title)
        if not data.empty:
            # 运行词库分析逻辑
            analysis_res = analyze_features(data['review_content'])
            
            # 绘图：Pain Points vs Highlights 
            fig = px.bar(analysis_res, x="维度", y=["Highlight Count", "Pain Point Count"],
                         title=f"{title} - 优劣势分布",
                         barmode='group',
                         color_discrete_map={"Highlight Count": "#2ecc71", "Pain Point Count": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示关键结论
            top_pain = analysis_res.loc[analysis_res['Pain Point Count'].idxmax()]
            top_high = analysis_res.loc[analysis_res['Highlight Count'].idxmax()]
            
            c1, c2 = st.columns(2)
            c1.success(f"🌟 最大亮点：{top_high['维度']} ({top_high['Highlight Count']}次提及)")
            c2.error(f"⚠️ 最大痛点：{top_pain['维度']} ({top_pain['Pain Point Count']}次提及)")
        else:
            st.warning("暂无数据")

    # 左右对比布局
    col_left, col_right = st.columns(2)
    with col_left:
        render_analysis_section(sales_data, "🔥 高销量板块 (Pain/Highlight Analysis)", "Blues")
        
    with col_right:
        render_analysis_section(trend_data, "📈 高增长板块 (Pain/Highlight Analysis)", "Oranges")

    # 底部展示原始评论预览
    with st.expander("查看原始评论数据预览"):
        st.dataframe(filtered_df[['sub_type', 'review_content']].head(100))
            
else:
    st.error("未检测到数据。")
