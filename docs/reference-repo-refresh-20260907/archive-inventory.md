# ZIP 源码归档只读清单（2026-09-07）

已合并既有14仓库、新增10仓库及补充的非命名匹配ZIP清单；范围覆盖这些下载记录中捕获远端分支HEAD可达的全部ZIP。

只从本地Git对象读取ZIP，在内存通过zipfile列目录并完整读取Python源码字节；没有解压落地、执行、导入或编译参考代码。源码已读取及哈希一致不代表算法已审计。

## 当前统计

| 指标 | 数值 |
|---|---:|
| repository_archive_pairs | 282 |
| globally_unique_zip_blob_sha1_count | 282 |
| archive_path_aliases | 284 |
| branch_path_occurrences | 316 |
| member_count | 289 |
| python_source_members | 283 |
| python_members_fully_read | 283 |
| python_unique_source_sha256_count | 283 |
| python_bytes_read | 32746082 |
| python_sources_skipped_or_failed | 0 |
| archives_with_errors | 0 |
| illegal_member_paths | 0 |
| symlink_members | 0 |
| non_python_members | 6 |
| encrypted_members | 0 |
| python_text_decode_failures | 0 |
| archives_with_duplicate_member_paths | 0 |
| archive_size_limit_skips | 0 |

## 非Python大成员与文件名说明

`jansenin-quantization-contest-work/contest_archive.zip` 中的 `example/mini_sample/attn.pt`（58,721,250字节）与 `example/mini_sample/linear.pt`（59,140,090字节）超过单成员32MiB上限，均明确标为 `skipped_member_size_limit_non_python`，未读取内容。Python源码越限或失败数量仍为0；没有ZIP超过总解压128MiB上限。

6个非Python成员包括3个目录、2个上述PT数据文件和1个Markdown文件，均只记录元数据。该Markdown成员名按ZIP记录的默认字符编码显示为乱码，原样保留，未推测改名。非法绝对路径或父目录穿越均为0。

## 分仓库统计

| 仓库 | repo+ZIP blob去重数 | Python成员 | SHA256去重源码 | 未完整读取源码 | 异常/错误归档 |
|---|---:|---:|---:|---:|---:|
| chiparon-HscCon | 1 | 1 | 1 | 0 | 0 |
| gitcode-quantizer | 3 | 3 | 3 | 0 | 0 |
| jansenin-quantization-contest-work | 1 | 2 | 2 | 0 | 0 |
| openaddr-hif4-contest | 52 | 52 | 52 | 0 | 0 |
| suibianll-AI-Contest | 1 | 1 | 1 | 0 | 0 |
| test | 54 | 54 | 54 | 0 | 0 |
| klompn24-crypto-2026_compatition | 170 | 170 | 170 | 0 | 0 |

## 限制与证据

- 单成员解压上限32MiB，单ZIP中央目录声明总解压上限128MiB，单ZIP实际源码读取总量上限128MiB；超出成员保留元数据并明确skipped。
- 对全部成员检查绝对路径、Windows盘符和父目录穿越组件；只记录，不解压。
- Python源码后缀为.py、.pyi、.pyw；二进制pyc与非Python成员只记录元数据，不宣称CRC或内容已读完。
- 每repo按Git ZIP blob去重，保留所有已知path与branch；JSON的branch_path_occurrences按清单固定commit逐条核验，不用集合笛卡尔积猜测。
- 完整源码读取计算SHA256，并由zipfile在读到EOF时核验CRC；只提取hif4 def名称字面匹配，不证明最终绑定、接口正确性或算法优劣。
- 未遍历未下载对象、ZIP历史版本或嵌套归档；不递归克隆gitlinks。
- 输入清单路径与SHA256、每ZIP的blob、路径、分支、成员总数、Python路径/大小/SHA256/读取状态均保存在[archive-inventory.json](archive-inventory.json)。

## 逐归档索引

| 仓库 | ZIP路径（相同blob合并） | 分支 | blob | Python成员/已读 | 状态 |
|---|---|---|---|---:|---|
| chiparon-HscCon | solution.zip | main | d80d1efba57ec52172b5ad2c1061c9938711a468 | 1/1 | inspected |
| gitcode-quantizer | candidates/submitted/c011-linear-pack16/solution.py.zip | codex/r4-block-task-diagnostic; hif4-fshq | 6fffe90af007faafa2a39c21e79f8fb39cba4edd | 1/1 | inspected |
| gitcode-quantizer | candidates/submitted/c012-sequence-k-center/solution.py.zip | codex/r4-block-task-diagnostic; hif4-fshq | 9afcb5746b80e2f026b163ba30c1a939b561d161 | 1/1 | inspected |
| gitcode-quantizer | candidates/submitted/c013-candidate-cost-cache/solution.py.zip | codex/r4-block-task-diagnostic; hif4-fshq | 8165d644715a8fa0375e606e307f449a30de0cd8 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v1.zip | main | 32cda86e7f69ff8cdb864a69e1621d2ea65879a5 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v10.zip | main | 6c486bcc12b9de28bdb28b16658bd7257a379ecc | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v11.zip | main | 83c4a5ce561ac26d07d3a1ff138e0efbefe5a5f0 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v12.zip | main | 6f6bd99d5287a3961a4cbd3ed02f403fe86ebca3 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v13.zip | main | a9832657c18c77432f8645d96907375a52eb4ede | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v14.zip | main | 638d82c642295242482fa7449a134743261e294f | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v15.zip | main | ecc8eecaf5611f269dafba72d990bf5663726a0f | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v16.zip | main | 4c4c771e9b8f6198949717340ebd92744aa43bd7 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v17.zip | main | 95741a69d2a1b4dcc7f3b3bdfec90d75f2c03bf0 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v18.zip | main | 13a2971efcc6d87856d2da4c903fecf732569e87 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v19.zip | main | 52de0ec6aa1bc2615470bdbacf13d9b9f95a2bcb | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v2.zip | main | dacbe790a9fb94deea1f31a8d83264725b8c3028 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v20.zip | main | fc830ec0cb4b9ec0cf7b3b72836b64e4b79d902f | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v21.zip | main | a119fd36fd8344ddb008decdc324a72a7feabc77 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v22.zip | main | 1f4c4196670167e4e3fa6d4f515a3281edaf3182 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v23.zip | main | 847ed9268f0202f4a8a29c1087728e17b1ea305a | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v24.zip | main | 9ff816f0e476ddad4751b1dc7b008c97b49b8c3d | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v25.zip | main | 0b2d149b85badcdfb232671856843ffd7ee66b0d | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v26.zip | main | 7341c620e8242f8838f8557876b4eb0320e6b956 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v27.zip | main | fc370fd29290f53660f46120803c4177c4537670 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v28.zip | main | 16995ea0aba13238df18facd164317d42e586610 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v29.zip | main | f4d52e48a3877128dee4dace54882be6590d37e4 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v3.zip | main | 5fb7c0b4f1617e64df528f1150243388b32f209c | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v30.zip | main | 086e7ad18bf8d3ea14272c2ff142bc37d87f9254 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v31.zip | main | 5b5c445bb737ba13bd488f30b8fb10720e7343c2 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v32.zip | main | ff822105b2fc4e788fa0ff487818894eebc9b5c4 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v33.zip | main | 2c6d6c78a030b479d7aebf064a2ed55977996ac8 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v34.zip | main | 31bc9e7263a20f72bd46191dd580d3e95ee516bc | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v35.zip | main | 3386bf844b7d621d63926e7face718e5e72ecf7b | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v36.zip | main | 5b64a5fa3d55cbf25472d791e514ddd25a1df9b5 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v37.zip | main | ce603bf51c1a928e9284921f71e4a11d10e6ea3d | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v38.zip | main | e42b29a969ebbd8ace1393b70c17dbccded57820 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v39.zip | main | 3a9814d7241029db5f7796b6772c89efb024a51b | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v4.zip | main | 9e4c6149e058b8e7b2a1fe8f8f796c3a0ce20e82 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v40.zip | main | c90f6524b9f32e6cc00ad153863fa3fb9c3e2d62 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v41.zip | main | e195ff452105d47da00ca2da3e02ce10d5e2b410 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v42.zip | main | 9a45156ba3cb944356290c4caef4e3572ea71b54 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v43.zip | main | 477ded23d14cdce0c10f60ba6f7c71690644e08d | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v5.zip | main | c96acea55996ce01277e895182643e50a9c09feb | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v6.zip | main | 243afb146d3b8c7c1f7afeb4d687823fe3a95ddd | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v7.zip | main | b7bf5254e85bc662717bbed642296a60338e8681 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v8.zip | main | 4828a4ddde6475a1980c47dd7d6df853e9fa5c19 | 1/1 | inspected |
| openaddr-hif4-contest | dist/solution_v9.zip | main | c8cc92cfeda1ff620276c1859bfb241dbed75e55 | 1/1 | inspected |
| suibianll-AI-Contest | solutions/20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA/solution.zip | master | f27a4b7637465b55603cbb1438f5aaea3cd1e3b9 | 1/1 | inspected |
| test | solution/submission-attention-compensated-v2.zip | base_1.7 | 2976965e7570ba94162b6892e6fb320d46c0a3cb | 1/1 | inspected |
| test | solution/submission-attention-joint-v1.zip | base_1.7 | e3e2fb6e42149a7c886dffa31f06ea581f5a4679 | 1/1 | inspected |
| test | solution/submission-attention-medium-v5.zip | base_1.7 | e4bf03ee59d8d36d9356137b4641d13206a2e3f1 | 1/1 | inspected |
| test | solution/submission-attention-native-v4.zip | base_1.7 | 6ab50c236810d87cad335d271d263ac712811bc9 | 1/1 | inspected |
| test | solution/submission-attention-wide-v6.zip; solution/submission.zip | base_1.7 | 19f4c12dd43647a776cd7893198d8754b6458eea | 1/1 | inspected |
| test | paper-solutions/attention-calhessian-linear-inputonly-safe/submission-aonly-v4.zip | experiment/legal-inputisolated-v4; experiment/legal-inputisolated-v5; experiment/unrestricted-headadaptive-v2 | dfae7d25953c4e5c21a9f982aeb215fac2bee2f0 | 1/1 | inspected |
| test | paper-solutions/attention-calhessian-linear-inputonly-safe/submission.zip | experiment/legal-inputisolated-v4; experiment/legal-inputisolated-v5; experiment/new-paper-implementation; experiment/unrestricted-headadaptive-v2 | 0edda1f8731d919863704e33fb0e829533738263 | 1/1 | inspected |
| test | paper-solutions/attention-calhessian-linear-inputonly-safe/submission-v5.zip | experiment/legal-inputisolated-v5; experiment/unrestricted-headadaptive-v2 | d1479ad7e0a4081b4057fa9d5cf3f493b99b1d16 | 1/1 | inspected |
| test | paper-solutions/attention-quotient-givens-linear-inputonly-safe/submission.zip | experiment/legal-inputisolated-v5 | 403624086c9fb7745d91b42a56a5a2094a36bcb0 | 1/1 | inspected |
| test | paper-solutions/runtime170-logitbasis-headadaptive-36k/submission.zip | experiment/legal-inputisolated-v5; experiment/unrestricted-headadaptive-v2 | 9cef5c8eba8db9edb05da68b4e373a0ba0733cb6 | 1/1 | inspected |
| test | paper-solutions/target20-attention-affinefit-safe/submission-attention-quotient-givens.zip | experiment/new-paper-implementation | 05ccd33bc0ad6346cd19acc90667e3554e9e073d | 1/1 | inspected |
| test | paper-solutions/runtime170-logitbasis-headadaptive-v2-36k/submission.zip | experiment/unrestricted-headadaptive-v2 | 34bbb10057f5ddd7ce75a444c7a84ca007e4ef14 | 1/1 | inspected |
| test | paper-solutions/combo-unrestricted-attention-kexact16-vhigh3-vexact4/submission.zip | experiment/unrestricted-highscore-v1 | 20b9fd486afb603ce036864110bc06d7079687c9 | 1/1 | inspected |
| test | paper-solutions/combo-unrestricted-attention-kexact4-vexact4/submission.zip | experiment/unrestricted-highscore-v1 | 83861b158a8bd5b64148561c5341213ac2a2df31 | 1/1 | inspected |
| test | paper-solutions/combo-unrestricted-attention-psot2-mincal2-kexact16-vhigh3-vectorized-vpostpair-fast-vexact4/submission.zip | experiment/unrestricted-highscore-v1 | 022517532464990004d998e8f2e4607133b63410 | 1/1 | inspected |
| test | paper-solutions/combo-unrestricted-attention-vexact4/submission.zip | experiment/unrestricted-highscore-v1 | 2a7cb900e455f4859c7169dcd6511df12b1b5605 | 1/1 | inspected |
| test | solution.zip | experiment/unrestricted-highscore-v1 | 358488deb1c82a882138368c2a1d3022cc904f17 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-10/solution.zip; archive/submit-9/solution.zip | main | 7bb930c837b7a03bb95e8c3bcce8ece5508a3e8b | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-100/solution.zip | main | f728ef1626a9e5cd00e23ce3039c0006ade723d1 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-103/solution.zip | main | bde01226d9caadbf296baf79ad243103b2925cd4 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-104/solution.zip | main | 557e34ad1f9fdcf13d24cbe45e19c011d153b1a3 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-105/solution.zip | main | cc4b08ed55800067fce712eb905a6ed49897550b | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-106/solution.zip | main | bbfa994c14909c907cb350b1514f47a10054aae4 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-107/solution.zip | main | 46175fa7f6638f8531c5da8331e739b8c507c1b2 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-108/solution.zip | main | 025cc46eb09870ffbb066d5e73311ecb16677e41 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-109/solution.zip | main | 5c57bb253ee878b2cdb537933a70e7c10162e743 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-11/solution.zip | main | 185f418193dd7b5c170e3e150eb75b3acbc5e6e2 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-110/solution.zip | main | 01834266dbac4382d4def6b8706ac8f2cd8ccf42 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-111/solution.zip | main | fb7802e8bec243679a6516419057bf0338ec7fa2 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-112/solution.zip | main | 81763816294c0308fb02fd3cac3a174ae0da2225 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-113/solution.zip | main | b3a65d077209f2414618d0d9e977730e3fa7deb1 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-114/solution.zip | main | 79e4a438e938c449c5908d937fb34cfeac203c52 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-115/solution.zip | main | 859ed310efe643018200b9884cbb1aa251409546 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-116/solution.zip | main | 2da1d41dc21d311ae0d1e9f3e0aec6e2f9ecc182 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-117/solution.zip | main | 204b66a06e5961c720c2974e80c3d66afc4b57ea | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-118/solution.zip | main | 5baa0750804b3b60facdae87f7775de2eeddedca | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-119/solution.zip | main | db9c269bb15225cc61959e53bdb1d5fc9f3b9ded | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-12/solution.zip | main | ae3097f5af4b9b5dc57e0ec2124a50f57b63fa6d | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-120/solution.zip | main | 142de3a7248c8348f56609322cb4a3f54d9c669b | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-121/solution.zip | main | 550c6b0f51a89ecf958b65794f48979852632c9f | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-122/solution.zip | main | 6c46a224e2ec0cf6d13f67faf7640cea2ed66a3c | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-123/solution.zip | main | afe8afea912250ad55d9bb49b86624300139932d | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-124/solution.zip | main | fd6c49189c3d9ef02c83c7c7fc2b6f9830aee1cb | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-125/solution.zip | main | ba35c15cdbc56543667041d125cb8aa23517d413 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-126/solution.zip | main | d883d9cef997384ef3c3cf48442810c1b41b7116 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-127/solution.zip | main | c54ab582e7b90ee412542a7d15de3a042e0e6d21 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-128/solution.zip | main | fd3d6cc7b4b25f2c1fc19da761d16c73be749ab2 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-129/solution.zip | main | 60940afddb4c1ded66de30e2c1c072fa90736eba | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-12b/solution.zip | main | 87e664201cb49e21874d9881dc942c71fed15558 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-130/solution.zip | main | b8a5f863f9a5a8513fcd7adaae39a4cb7d4b910f | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-131/solution.zip | main | 61891c14c0610f256edea49b64bc99e3c147bd9f | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-132/solution.zip | main | 4dc1456a591a61fbf4f10f78344ac1c7ff43d21c | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-133/solution.zip | main | 294c3a6c5356ba9145f028959b8d5089c674ff97 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-134/solution.zip | main | bf3c8489f8ecf227d403e3c65395add72c2db963 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-135/solution.zip | main | 8a2f35479cd7cce6b0fd5149a6098ec39910c482 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-136/solution.zip | main | 290adf2e6995b5af4fc7aeb710c032f28ba66b43 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-137/solution.zip | main | 978e40ed09697fd50b2936d7589abfbc95a49d25 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-138/solution.zip | main | efcb1a313ebf7b12901af5ade99d1205c477b11b | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-139/solution.zip | main | 84ef3c17d19883bb4d9ac25a737e2db0742c92cf | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-14/solution.zip | main | 8d899b0f66e96e54f5fde4f1cda23c2d6edca1b5 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-140/solution.zip | main | b6eefe44438ee1de654da23be7c2615e6d14e411 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-141/solution.zip | main | 472598877be7e253e17bef53f4d83143cede027c | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-142/solution.zip | main | 5f587547b597dc086c0a52393d964e643ad7a94e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-143/solution.zip | main | 6a9c1fec304179aea13ddb3e997f6f8f749685a1 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-144/solution.zip | main | 737026ebc69188d4404e286f43a108645aaa462e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-146/solution.zip | main | f94ebdde8947ea3c55d13fd26a07aaf1e8a173d4 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-147/solution.zip | main | a84034e5d55b50388f81bd4d4f2ece44e3afef85 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-148/solution.zip | main | 2f9ed5c35441ee74b3139d7d18693461ebfa4e1e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-149/solution.zip | main | 72bb9c589a4b81d798d8d25b2ef3a9c18580e342 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-15/solution.zip | main | 6c539343196ccc3bcf0ea7f2475c621fc6e33282 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-150/solution.zip | main | f3b4eefabafda96c031f917572daf690d358abcd | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-151/solution.zip | main | a51cd3ae50606645cf7475a987bbd9ad5a9f65c0 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-152/solution.zip | main | c1abbd137b6dac4f9b10710c188675dfc217213c | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-153/solution.zip | main | 603a155259b91d424d03301124ac940d09558c36 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-154/solution.zip | main | 904f8abfd169aebc502bbfd3a82bbbef851f909f | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-155/solution.zip | main | 816066dac1dd23de41d15d9feb664bd3537ddd5a | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-156/solution.zip | main | eecc589da1fc40f87f1b3ef23073a839036eca94 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-157/solution.zip | main | b07bb9a87cc03121a730fa29b5e5885b9fd23f5e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-158/solution.zip | main | 3eeae5093933e75b652f7aa90c09c293a9b69200 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-159/solution.zip | main | afeffa83f8fa53bac355d56e626afaada3458fec | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-16/solution.zip | main | 0a870787cc3a1b2b541ac447134b2a03ec4881f8 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-160/solution.zip | main | 6fad165c9c46b289cac5846d9e0e57eb24322205 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-161/solution.zip | main | 91e46eba9dad8c6df32c3e419f5904554cb75cf4 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-162/solution.zip | main | 6e6f77ded27bc5c3d3cf2becf3d57c85cc739cee | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-163/solution.zip | main | 4052571305dbed24b5e63790782319618f915153 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-164/solution.zip | main | 3470aefbdfe2b013fffeed0646ff784a4d4cab55 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-165/solution.zip | main | 605647830c6294e4b55fcff42740269e0ff5d902 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-17/solution.zip | main | e934790b7be16329564ecf8a9ebaa1428242096e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-171/solution.zip | main | aa4cb6dc2c9b5fbadea88257514f1268c61ae242 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-172/solution.zip | main | 07cf8faed981d74c3feba58f744c1f34de6ce82a | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-173/solution.zip | main | 7e780c06b660c9c3744e1073c0f0059c3c0b7859 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-174/solution.zip | main | 6c11da7b1beb2d0fb8734da425421058b1cef4d7 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-175/solution.zip | main | 0f7aaeb3696827697df285f2c64c146f6a9f88f1 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-176/solution.zip | main | ca81e7301c42566b4692a795f928edda99c097a9 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-177/solution.zip | main | 7ec841a320caf98c5994572e3d2afb56ae83aeca | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-178/solution.zip | main | 3697f5100d9e871d0ac157c8ffb217f4f9b9d581 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-18/solution.zip | main | e67ae81630fba5a495f2c5b50fcb48cc44002426 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-180/solution.zip | main | 9e0b42e2499ddebcf3ca742198540648231d3571 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-181/solution.zip | main | c48e5b1658d387e8a7f73b5a6c23067312d5b866 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-182/solution.zip | main | 3c4093499f0d2ad1f8d89eee85d172e42a13576a | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-183/solution.zip | main | 871ed4e1a7a33f53f406890aefcf657b19ee6ccd | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-184/solution.zip | main | 9564371d787790fc3b91687c9aa3d2f94da0ae69 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-185/solution.zip | main | e59bee225b0ed3b4c6c4f22c991891b0645ff233 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-19/solution.zip | main | e14dfbf2618afc8006335e31f1609e78e59416d2 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-20/solution.zip | main | ee3bb59b4197fe5f0954ec0426498404f4177020 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-21/solution.zip | main | 96699d4b6b065617b3f6499d23a17d9f1f6d990a | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-22/solution.zip | main | 29e26475968b5c10ca71d6e10aa20d37cfaaade6 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-23/solution.zip | main | 3453620364ce49f859574b25ca092c628e0c998c | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-24/solution.zip | main | ed64655f63784f37177d3fdd29ee6bc482974d63 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-25/solution.zip | main | 664756144c741d9ac7ac13447a307d4057283e42 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-26/solution.zip | main | d4c4d5a3f9d0666a1126347486e84fb8e492729d | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-27/solution.zip | main | 8d0f0268117381098ee104c929f2e2db90cd10ba | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-28/solution.zip | main | 0f06f4604f3f44e19b3dab042be222c03c0a70c0 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-29/solution.zip | main | 56dcfc121d79f1fcdbabcb056df7b327e50eb643 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-3/solution.zip | main | abb202f4f995d77726e652ab9c3f90eeb3524db8 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-30/solution.zip | main | 910f32bc20e8b95d02dbd9da57a3575de8632570 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-31/solution.zip | main | e44d08d89189cd933a250e4d9527bc824158c853 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-32/solution.zip | main | 4abc3b71543b2c1d17742c06e922e11b6d9fbc9c | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-33/solution.zip | main | 3f11c1f7ab06f6ca9d4384aec3f49e9b0a529f90 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-34/solution.zip | main | f24ef64341950e038c13f096476d2ffefb43faf2 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-35/solution.zip | main | 8cf9f258eaadf1e095b96ed137ae432011a971da | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-36/solution.zip | main | 0c2a7f1bbf501067fd66677c8236a5ad1dcd1a10 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-37/solution.zip | main | eda479842b6b1c94eec5d826c11e61146cf2e2d0 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-38/solution.zip | main | 2bf9c3e60ad9180a58d3ccff3dc0e8c655e0a493 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-39/solution.zip | main | 0a5ba7ab6c367f11965af2c119e766af177a911d | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-4/solution.zip | main | 32a5e8e5536e7e5899e5e3d34dc5e592b754c3f6 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-40/solution.zip | main | f2126bd1c4cca97eb235fdde208a5e6eb2be23be | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-41/solution.zip | main | 90a0d1aeb44cf4096a6ba9084699fefda415c704 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-42/solution.zip | main | 6e6a7027f188163fda2816a78926a88c3f51e104 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-43/solution.zip | main | 738c38ad44865c316f7e4eff651bff903fff0a69 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-44/solution.zip | main | 63b1bad482fc8d7b3038b44cd5b99a0a691f34d4 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-45/solution.zip | main | c3200eafa03f886874c3df9c9906300e16b7ff3f | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-46/solution.zip | main | a0089839ee1c4cee5d89a54aa0f32609d5919d08 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-47/solution.zip | main | c04329fbc743804236a474de0db93e2ea191491d | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-48/solution.zip | main | a125145251d5e07de5a7345c696cccf836706b20 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-49/solution.zip | main | c8dea436b282d2b5723cbf81452a8bec20df87f0 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-5/solution.zip | main | 796a52cc50e87b15a8b5fb1a1256a22ce3c77b5f | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-50/solution.zip | main | 249fdc807b18c467ea587dffa8960fb3322bc597 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-51/solution.zip | main | 56aed0ced03db0009b865419a23dd749f0349476 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-52/solution.zip | main | 93970bbf7e8831bfc7cf48aedbea87da90628296 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-53/solution.zip | main | 7b6a3cc3e1387024e8b6943497c8513b7df138a0 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-55/solution.zip | main | 0d90d188ec8906619c6584ad55abce6be7bb462d | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-56/solution.zip | main | 4a89f7ea004a14171ebba24e0abf0d9ea7f75a16 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-57/solution.zip | main | 62229f77557c059d7c3e9651ef3fecd1150fa0cb | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-58/solution.zip | main | 30110838b010371a5e2b44a1999cd4f5e3570ea0 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-59/solution.zip | main | f6e5588b012a3e37265e565bd696515c75539f2e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-60/solution.zip | main | 4efedb312de506a15bff54b9d1ccd3903601a61e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-61/solution.zip | main | 33de8cfcae50335e67cea950ece195ca6302bfa5 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-62/solution.zip | main | 226f7bc582fbac12039c161f6bb9a4ec77b29ef4 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-63/solution.zip | main | e34633bc35434fd04febbd2e9634ef304fca631d | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-64/solution.zip | main | 114a58e781bbfbcf5af309a6282129c455bf6d92 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-65/solution.zip | main | 5f71aa75e1623791ddc5db10aaa4887edfe6fedf | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-66/solution.zip | main | 2767ef87e8deb1d31a0ec269d38cb776af6c4f7e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-67/solution.zip | main | fca00016984c4e3a7df4790093b3ffa49d584245 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-68/solution.zip | main | 6e4d5fd7de5f1b1bb9c1fcbec8349611b023d919 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-69/solution.zip | main | aefae232b242300aac94521bb89864cba4a21846 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-70/solution.zip | main | 285691dbce19718667bf408ab5cd7df8fcd50c9e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-71/solution.zip | main | 5aa9248a951c1f88f9d914ff4acfa30a428b9ff0 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-72/solution.zip | main | 8ef3fad882da1925b95ad98c857111b4505fcd36 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-73/solution.zip | main | aeadf01b92ccae77edc10d1ad6b802154a706a4f | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-74/solution.zip | main | 2d8e5ec1795732de771217e6a6c78d705099239c | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-75/solution.zip | main | c90bac322bb423a5f5697182bbc4eec01a712e07 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-76/solution.zip | main | c5c6e8e9c4ed587042fd33a8096bfc49207183cc | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-77/solution.zip | main | 885e8b9784c21ae382ed7db64cbb451a1015f6a2 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-78/solution.zip | main | e0a24997807c9fe3c433aed204bec3fe336627f3 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-79/solution.zip | main | 6d07dfc1b0dce9c2eaa50f0401d6ed077f5c43d8 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-8/solution.zip | main | 6f1cee77e02f3aceb2a67e19e9087053fbef4fb2 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-80/solution.zip | main | e392f46cfe8499ddab2935958a3838ee8bc8ea00 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-81/solution.zip | main | 339143a27d1c7025317131f7ecb3cb737fd32d4e | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-83/solution.zip | main | 4994bd649a56154959aa7104f923e64f762df796 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-84/solution.zip | main | e372414633c11e9401e255043ca223a155957a56 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-85/solution.zip | main | 7de578d2250135622b0b5230f3838d01ac5060ca | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-86/solution.zip | main | d18aa373f221b525ac5981a2b45c5acdebb7cce3 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-87/solution.zip | main | 1bdbe4f582fd8ccf5472ceaa623189674f218f39 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-88/solution.zip | main | a126321c8d9ec1ae9112aaac7ba740f9902d2fab | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-89/solution.zip | main | ba52ec527d95ef997fed802bc4a2f060b683a94a | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-8b/solution.zip | main | f9ef8605465d31de5f3a4240a8712280df3e6d18 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-90/solution.zip | main | 3fff04625a8d43b019da22e7f4a4f641b6d8b8a4 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-91/solution.zip | main | 5ad00d90eb6211ebef19d9ca6cbdc281cfdd8556 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-92/solution.zip | main | 812f0b0c58caa01db5d5c0f886bb0daa50d8ad5c | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-93/solution.zip | main | e3bad8af5ff6ed288210918ebce76aa8b4f7610b | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-94/solution.zip | main | 3807bc99216e6e6da4c81bdc74b562ace3ef377f | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-95/solution.zip | main | 897cdc6bc9ec0db06ae97486073362c540e58faa | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-96/solution.zip | main | 0ab50631e8a338daa2557b4bbec50efa75d16075 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-97/solution.zip | main | 9364f0f93743880bd8604813a77f06506c35da34 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-98/solution.zip | main | 9df841fa221463910574a4899cbd6d40ccab8821 | 1/1 | inspected |
| klompn24-crypto-2026_compatition | archive/submit-99/solution.zip | main | d2552ae8ce15dcc4039c0fb862c93d425638d4f2 | 1/1 | inspected |
| jansenin-quantization-contest-work | contest_archive.zip | main | eda6a310b13c0a5a0d806b1b7d1008098fa3f847 | 2/2 | inspected |
| openaddr-hif4-contest | dist/probe_ablation.zip | main | 7c896df888e83f6be18496bef1898cac489d8587 | 1/1 | inspected |
| openaddr-hif4-contest | dist/probe_alg1.zip | main | c65af42f0fe88a08d3a58b6aadeff9ea5b42a321 | 1/1 | inspected |
| openaddr-hif4-contest | dist/probe_c4096.zip | main | a14ac2f014f11242f95869a8ebd173e39e1db3df | 1/1 | inspected |
| openaddr-hif4-contest | dist/probe_carry.zip | main | e55de4f9224a59274f527799534ec76881da098d | 1/1 | inspected |
| openaddr-hif4-contest | dist/probe_carry2.zip | main | 61242fee055649eaaa99cb4a07a4548f484c92ad | 1/1 | inspected |
| openaddr-hif4-contest | dist/probe_carry3.zip | main | e782ad0ea1b0f7a7572f59718468f3c082be5ddc | 1/1 | inspected |
| openaddr-hif4-contest | dist/probe_cband.zip | main | 89ea2fbfc99025f97cb15040630736a48512e797 | 1/1 | inspected |
| openaddr-hif4-contest | dist/probe_linear.zip | main | e5f1fa528208e5525480e0692605dd57e85c7f81 | 1/1 | inspected |
| openaddr-hif4-contest | dist/probe_v9_linear.zip | main | 39aad371dd25ae4d758694dbe8e641d95c102e15 | 1/1 | inspected |
| test | legal-solutions/zips/legal-rht-adaptive-v1.zip | base3.07; base_1.7; experiment/attention-fisher-objective; experiment/attention-learned-rotation; experiment/block-hessian-objective; experiment/joint-output-optimization; experiment/legal-inputisolated-v4; experiment/legal-inputisolated-v5; experiment/legal-tensor-recon-v2; experiment/linear-residual-fit-v2; experiment/new-paper-implementation; experiment/non-rrs-paper-baselines; experiment/output-aware-rounding; experiment/paper-solution-suite; experiment/pca-hadamard-attention; experiment/qwen-channel-balance; experiment/smoothrot-19k; experiment/smoothrot-32k; experiment/smoothrot-32k-v2; experiment/soar-hif4; experiment/unrestricted-headadaptive-v2; feature/hif4-solution; main | ceba4208f55f8cb607321db832ab0d52fbc7c0c0 | 1/1 | inspected |
| test | legal-solutions/zips/legal-aonly-smooth-rht-v2-1024.zip | experiment/legal-tensor-recon-v2 | b211c2c3d28b10ad2867902a72002fe110f6f8c4 | 1/1 | inspected |
| test | auto-commit/data/archive/trial_001_score_8972.133354277346.zip | experiment/unrestricted-highscore-v1 | f019fcfdfd5762170a94215d79c5f7f84c61c3a3 | 1/1 | inspected |
| test | auto-commit/data/archive/trial_002_score_9545.285635925295.zip | experiment/unrestricted-highscore-v1 | f392e3bd42b141056daaff8a2cff9ef27dc7ff4a | 1/1 | inspected |
| test | auto-commit/data/archive/trial_003_score_9545.285635925295.zip | experiment/unrestricted-highscore-v1 | 6ee159bbcbcb3efae7f8d5123c6a63102e1eab52 | 1/1 | inspected |
| test | auto-commit/data/archive/trial_004_score_9545.285635925295.zip | experiment/unrestricted-highscore-v1 | 5c892d67e6b9b7a412b65383a642c3b876a9aa6d | 1/1 | inspected |
| test | auto-commit/data/archive/trial_005_score_9731.699839527024.zip | experiment/unrestricted-highscore-v1 | fd83380eebd7f0cc97255f295be9d25655168cf8 | 1/1 | inspected |
| test | auto-commit/data/archive/trial_006_score_9731.699839527024.zip | experiment/unrestricted-highscore-v1 | 3c106092f63b08e91367d49a599c6b4813136694 | 1/1 | inspected |
| test | paper-solutions/zips/awq-fast.zip | experiment/unrestricted-highscore-v1 | fbce2bc77edca0a3f6b075964ce93275473e6ccb | 1/1 | inspected |
| test | paper-solutions/zips/combo-cross-adaptive.zip | experiment/unrestricted-highscore-v1 | 46444fec00245299e715daf605db6341a6d4b6e3 | 1/1 | inspected |
| test | paper-solutions/zips/combo-cross-refine20.zip | experiment/unrestricted-highscore-v1 | 33c1082da58a20cc36426982e4c4376c8a00a85b | 1/1 | inspected |
| test | paper-solutions/zips/combo-cross-refine4.zip | experiment/unrestricted-highscore-v1 | ab1fec75824e3b7d6b933eae72eee18e82ffe916 | 1/1 | inspected |
| test | paper-solutions/zips/combo-projection-100.zip | experiment/unrestricted-highscore-v1 | 61b32d3b136b8d0923cfd43e1d6ba8d19ed43295 | 1/1 | inspected |
| test | paper-solutions/zips/combo-projection-refine4.zip | experiment/unrestricted-highscore-v1 | 98654f300cee410378dec7bec83831d419041883 | 1/1 | inspected |
| test | paper-solutions/zips/combo-projection-target-group128.zip | experiment/unrestricted-highscore-v1 | 2e3ea138d47982535dccb3ae97820c856ca51a97 | 1/1 | inspected |
| test | paper-solutions/zips/combo-projection-target-refine20.zip | experiment/unrestricted-highscore-v1 | e92408d6bf39399b50172b638f34010973ca78e9 | 1/1 | inspected |
| test | paper-solutions/zips/combo-projection-target0-group96-batch8-scalar1024.zip | experiment/unrestricted-highscore-v1 | bf3cc8dc3e302ced4a97c11aad513c27348e09fb | 1/1 | inspected |
| test | paper-solutions/zips/combo-projection-target8-group132-fast.zip | experiment/unrestricted-highscore-v1 | ae442943ac47e3dd96d03e220218a2813f00c0b5 | 1/1 | inspected |
| test | paper-solutions/zips/combo-projection-target8-group136-batch4-scalar512.zip | experiment/unrestricted-highscore-v1 | 1e89276d7c76a523f25b5f0963f436100e6f0e5f | 1/1 | inspected |
| test | paper-solutions/zips/combo-runtime-khessian-efficient.zip | experiment/unrestricted-highscore-v1 | 54c063299c968689fe3c8c588776af5225545770 | 1/1 | inspected |
| test | paper-solutions/zips/combo-runtime160-fixed-attn4.zip | experiment/unrestricted-highscore-v1 | 5205dca93455286bbd2ae76a0125f3d8a25364b8 | 1/1 | inspected |
| test | paper-solutions/zips/combo-runtime160-online-state-factor-fp16.zip | experiment/unrestricted-highscore-v1 | f615587e213b7beea5d5462fa2b6cc301a0022df | 1/1 | inspected |
| test | paper-solutions/zips/combo-runtime170-ood-guard.zip | experiment/unrestricted-highscore-v1 | 02cab6b4c9e1e36e1294f3522894a03ea0b3b0e1 | 1/1 | inspected |
| test | paper-solutions/zips/focus-dgs.zip | experiment/unrestricted-highscore-v1 | 973736a198abeec793d1dd2e9bee29be62f57825 | 1/1 | inspected |
| test | paper-solutions/zips/infoquant-fit-weight-fast.zip | experiment/unrestricted-highscore-v1 | 2a1683e78096fa5bec7cfc2f50c89bf1c4a4ff3f | 1/1 | inspected |
| test | paper-solutions/zips/infoquant-per-layer-alpha.zip | experiment/unrestricted-highscore-v1 | fb88325cad53c0c4d9e047ecad5468d767ac46c3 | 1/1 | inspected |
| test | paper-solutions/zips/infoquant-rank1.zip | experiment/unrestricted-highscore-v1 | 38be68af2aa56243cfc7a912809f321eed3a9e50 | 1/1 | inspected |
| test | paper-solutions/zips/infoquant-rank2.zip | experiment/unrestricted-highscore-v1 | df73e44ef9ac7175d0f7903617c645e3101fbfe8 | 1/1 | inspected |
| test | paper-solutions/zips/let.zip | experiment/unrestricted-highscore-v1 | 5837fcb5574e1fff4bd8f292bd98c8cf8481485c | 1/1 | inspected |
| test | paper-solutions/zips/qqq-fast.zip | experiment/unrestricted-highscore-v1 | 1daace369906a55c2916c596ac3ee840b0bb4736 | 1/1 | inspected |
| test | paper-solutions/zips/rht.zip | experiment/unrestricted-highscore-v1 | 972de5a495d1df162b09e571cfc4cbee6210957a | 1/1 | inspected |
| test | paper-solutions/zips/rrs-fast-fp16.zip | experiment/unrestricted-highscore-v1 | c529f614d3a4cdb433ef2c36ef0f3f387f1c85f4 | 1/1 | inspected |
| test | paper-solutions/zips/rrs-fast.zip | experiment/unrestricted-highscore-v1 | e1974659de0b62ed70ede4b30fb84427769f56d8 | 1/1 | inspected |
| test | paper-solutions/zips/rrs-let.zip | experiment/unrestricted-highscore-v1 | c889607dbec27d16ae621de992b63fd1619a3ed2 | 1/1 | inspected |
| test | paper-solutions/zips/rrs-lite.zip | experiment/unrestricted-highscore-v1 | f275e0226d0bdced3a8971d3d9ce3e7cfc40d492 | 1/1 | inspected |
| test | paper-solutions/zips/rrs.zip | experiment/unrestricted-highscore-v1 | 64fd7b705262c212c2cfe65c8555ded79db35038 | 1/1 | inspected |
| test | paper-solutions/zips/spin-lwc.zip | experiment/unrestricted-highscore-v1 | 8f3f726047250f19aa1051d0a1b3688a74a2322f | 1/1 | inspected |
