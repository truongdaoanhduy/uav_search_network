# Kiểm tra lỗi UAV mắc góc — run 0b62f069

Ngày kiểm tra: 2026-10-11. Repo: /home/aduy/Documents/NCKH/uav_search_target.
W&B hiện tại: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/0b62f069
Project cũ uav_search_paper/uav_search_target đã bị xóa; các run bên dưới là khôi phục dữ liệu còn lưu, không phải train lại.
Mã nguồn nền hiện tại: 255b51c18115e2b444d7864c7c027d70eeaa3e04, cùng các sửa đổi trong working tree.

## Kết luận có bằng chứng

Policy đã suy giảm trong lúc train và tiếp tục ra lệnh hướng về biên/góc khi đánh giá. APF can thiệp vào các lệnh đó, nhưng không tạo ra mục tiêu tìm kiếm thay cho policy. Chưa có bằng chứng đủ để kết luận cần thay APF, thêm LayerNorm hoặc tăng hệ số reward để giải quyết nguyên nhân.

Lỗi implementation đã tái hiện và sửa gồm temperature vẫn bị Adam momentum cập nhật khi không có lựa chọn discrete; metrics entropy DDP chỉ phản ánh rank 0; kiểm tra temporal coverage chặn cấu hình khởi chạy mặc định; CPU visualize không phục hồi được các cấu hình checkpoint lịch sử; việc liên kết run nguồn làm fail quy trình khi project đã bị xóa; và mask GPU vẫn cho truyền report đã hoàn tất. Chưa chứng minh lỗi mask ở trạng thái report đã hoàn tất gây kẹt góc: quy trình GPU thông thường dọn report đó ngay. Việc hội tụ của policy sau train lại vẫn cần đánh giá riêng.

## Run gốc và checkpoint

Run đã finished, hoàn thành 20.000 episodes và 38.268.928 transitions. GPU training và CPU visualization Kaggle cũ đều complete.

Checkpoint artifact: masac-seed44-0b62f069-checkpoint:v0.
SHA-256 đã tải và xác nhận lại:
a6c84a96c831091a1aaa6ff378792582c77e4c6ce845c50b749c4ff44d2b7990

Evaluation seed 10044, một episode:

| Metric | Giá trị |
|---|---:|
| Return | -376,294 |
| Coverage | 2,389% |
| Target search / delivery | 0% / 0% |
| Coverage redundancy | 99,853% |
| APF intervention | 66,61% |
| Boundary APF intervention | 54,34% |
| Boundary clipping | 0,372% |
| Collision/separation violation count | 76 |
| Episode length | 2.061 steps |
| UAV cạn pin cuối episode | 6 |

Collision count là tổng vi phạm hậu bước theo mô hình: khoảng cách UAV/UAV dưới safety_distance hoặc xuyên cylinder vật cản. Không thể diễn giải thành 76 tai nạn vật lý độc lập.

W&B trả về 490 bản ghi training và 1.004 bản ghi episode có giá trị trong mẫu 1.500 history records. Đây là sampled history, không phải toàn bộ episodes. Trung bình 100 bản ghi episode đầu/cuối trong mẫu:

| Metric | Đầu mẫu | Cuối mẫu |
|---|---:|---:|
| Return | -34,61 | -329,74 |
| Coverage | 41,62% | 7,68% |
| Search | 8,58% | 0,28% |
| Delivery | 3,86% | 0% |
| APF intervention | 38,55% | 61,86% |

Bản ghi training cuối được lấy ở transition 38.113.280: alpha_discrete = 9.178,77; critic loss = 6.832.522.240; target Q mean = 264.928,28. Alpha ban đầu là 0,2. Giá trị hữu hạn không đồng nghĩa training ổn định.

Biểu đồ: corner-run-0b62f069.png. Một seed; không phải khoảng tin cậy qua nhiều seed.

## Vì sao mắc góc

Lệnh actor đã bị lệch mạnh trước khi APF xử lý. Checkpoint khởi đầu cũng ra các lệnh gần đường chéo hướng về góc. Vì thế hiện tượng không chỉ xuất phát từ renderer.

Phép thử chuyển động 80 bước đã làm trong phiên trước, từ trạng thái gần góc ở step 201:

- Giữ lệnh actor và bật APF: UAV vẫn ở gần góc.
- Giữ lệnh actor và tắt APF: vị trí cuối về (3000, 0), clipping khoảng 97,5%.
- Đổi sang lệnh điều khiển hướng vào trong, giữ APF: UAV rời biên, khoảng cách vào trong nhỏ nhất khoảng 529 m.

Phép thử này chỉ có chuyển động, không vật cản, sensing, network hay cập nhật pin. Nó cho thấy tắt APF không sửa được lệnh sai và APF vẫn cho phép thoát với lệnh phù hợp; không chứng minh APF an toàn trong mọi tình huống của bài toán.

Reward APF đã hoạt động: reward_component_apf_intervention của evaluation gốc khoảng -274,57, chiếm phần lớn return âm. Vì vậy chưa có bằng chứng rằng lỗi chỉ là thiếu phạt APF. Silent communication cũng cần đọc cùng mask: khi không có report/next hop hợp lệ thì silent có thể là lựa chọn duy nhất, không tự nó chứng minh lỗi networking.

## Cấu hình replay/warmup: giả thuyết cần kiểm chứng

Run gốc dùng 4096 env, replay 50.000 và learning_starts 2.000 transitions:

- Replay giữ khoảng 12,2 vector steps.
- Warmup chỉ tương đương 0,49 vector step.
- Thời gian hiệu dụng 1/(1-gamma) với gamma 0,99 là khoảng 100 steps.

Khi policy suy giảm, FIFO replay ngắn nhanh chóng thay thế kinh nghiệm tốt bằng trạng thái của policy hiện tại. Warmup gần như chưa có dữ liệu trước khi bắt đầu học. Đây là một giả thuyết cấu hình mạnh, chưa phải chứng minh nhân quả bằng ablation dài.

Mốc 100 steps trong validation là engineering guard. Nó không bảo đảm hội tụ hoặc bao phủ đầy đủ episode 3000 steps.

## Những thay đổi hiện tại

1. MASAC bỏ optimizer.step của discrete temperature khi tất cả agent/sample không có lựa chọn. DDP dùng MAX để mọi rank cùng quyết định, tránh cập nhật lệch hoặc deadlock. Test có Adam momentum từ batch trước, thêm kiểm thử hai rank thực.
2. Năm metrics entropy discrete của DDP được tổng hợp toàn cục: entropy, target, error, choice fraction và entropy bonus. Rank 0 không còn che mất lựa chọn/entropy của rank khác.
3. Replay/warmup lấy mặc định theo runtime qua OmegaConf oc.select, vẫn cho phép override algorithm:
   - Kaggle 4096 env: 409.600 / 409.600.
   - Vast 2048 env: 204.800 / 204.800.
   - Vast 4096 env: 409.600 / 409.600.
   - CPU debug: 50.000 / 2.000.
4. Kaggle lưu replay float32 trong host RAM và chuyển minibatch sang GPU; không giảm precision của replay. DDP chia replay theo rank. Chi phí lưu tensor replay ước tính khoảng 15,7 GiB toàn cục; job hai T4 đã đi qua warmup và optimizer trên runtime thật. Không suy rộng các số đọc host RAM thành bảo đảm bộ nhớ cho mọi container hoặc lượt train dài.
5. Có priming cuBLAS trước replay và giải phóng cache không còn dùng một lần trước pha policy. empty_cache không giải phóng tensor còn sống; nó không thay thế việc kiểm tra bộ nhớ thật.
6. replay_storage_device chỉ chấp nhận cpu/cuda, lỗi được báo trước khi launch.
7. CPU handoff phục hồi cấu hình lịch sử có thể thiếu trường mới; dùng default lịch sử cuda mà không sửa saved config/hash. CPU evaluation bỏ riêng temporal TRAIN guard; train mới vẫn giữ guard.
8. scripts/smoke_corner_recovery.py kiểm tra dữ liệu GPU → CPU replay → GPU, đủ 104 vector steps để đi qua warmup và có optimizer updates, kiểm tra hữu hạn và lưu checkpoint/result, đo VRAM/RAM.

Các thay đổi không thêm oracle target-distance reward, không tăng pin, và không đổi core stochastic actor/twin critics/entropy/target update của MASAC. Cấu hình reward/APF/scenario của paper được giữ khi đánh giá run gốc.

## Khôi phục W&B theo project mới

Đích đã xác minh truy cập và ghi trực tuyến: truongdaoanhduy-uit/uav_search_target.

| Run | Nội dung khôi phục | Link |
|---|---|---|
| 0b62f069 | 1.500 bản ghi sampled history, summary gốc, checkpoint, biểu đồ chẩn đoán | https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/0b62f069 |
| 28a7d6b1 | Video/PNG/metrics CPU gốc của run 20k, lấy từ output Kaggle hoàn tất | https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/28a7d6b1 |
| 806d03ea | 2.167 bản ghi console metrics còn tải được, summary, checkpoint trial | https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/806d03ea |
| eee3a6ce | Video/PNG/metrics CPU của trial, lấy từ output đánh giá thực tế | https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/eee3a6ce |

Cả hai training run được gắn historical-recovery/partial-history và recovery/fresh_training=false. Không có file lịch sử .wandb đầy đủ của các run này trong các backup đã kiểm tra. Không thể khôi phục các điểm metrics không còn lưu. Không nội suy hoặc tạo dữ liệu train mới để bù lịch sử mất.

Đã kiểm tra lại bằng W&B API: các run ở trạng thái finished, media MP4/PNG hiện diện, liên kết CPU -> training đúng ID. Digest file checkpoint được upload khớp bản local đã xác minh. Hai checkpoint có SHA-256:

- 0b62f069: a6c84a96c831091a1aaa6ff378792582c77e4c6ce845c50b749c4ff44d2b7990
- 806d03ea: be1164832d1aa55fe499f62bf6931b2580dee9258e99af8807ac71747fb75a10

Cấu hình mặc định và công cụ định dạng W&B đã đổi sang entity mới. CPU postprocess cho phép chọn WANDB_ENTITY/WANDB_PROJECT làm nơi upload, giữ namespace lịch sử khi tìm run nguồn. Nếu run nguồn bị xóa, media đã tạo/upload vẫn được giữ, báo source_run_linked=false. Việc khôi phục thủ công ở trên chủ động liên kết bản source mới chỉ sau khi xác minh SHA checkpoint. Có kiểm thử hết workflow bắt sai namespace trước sửa và pass sau sửa.

## Verification

- Các kiểm thử red/green của cấu hình/DDP, Adam momentum và handoff lịch sử đã thực hiện; xem lịch sử phiên và corner-validation-2026-10-11.json.
- Suite sau migration W&B: 247 passed, 0 skipped, 414,18 s, bật RUN_UAVNETSIM_INTEGRATION=1.
- Sau review bổ sung, 54 focused tests passed: destination/source namespace, CPU handoff/pipeline, replay precision và mask report hoàn tất. Regression mask trước sửa: 1 failed, 3 passed; sau sửa: 4 passed. Regression namespace trước sửa thất bại đúng địa chỉ; sau sửa pass.
- Bộ test cuối sau toàn bộ sửa source/mask: **252 passed, 0 skipped, 373,51 s**, bật `RUN_UAVNETSIM_INTEGRATION=1`.
- compileall và git diff --check thành công. Review độc lập không còn finding blocking ở ba sửa cuối. Review không chứng minh chất lượng policy.

Smoke cũ uav-corner-audit-2t4-smoke-20261010 đã cancelled, cuối log có vector step 100 và optimizer_updates=0. Không tính nó là pass hoặc suy ra chắc chắn OOM/cuBLAS từ trạng thái cancelled.

Smoke CPU replay mới:
https://www.kaggle.com/code/vuliu123456/uav-corner-replay-cpu-smoke-20261011
Complete và đã kiểm tra outputs: 2×Tesla T4, 4096 env, 425.984 transitions, replay/warmup 409.600, 64 optimizer updates; critic loss cuối 1,754; alpha_discrete 0,1962; metrics cuối hữu hạn. VRAM đỉnh 6.381 MiB mỗi GPU. Host MemAvailable thấp nhất 10.843 MiB; RSS đỉnh của child lớn nhất 10.055 MiB. Đây là số đọc host /proc/meminfo và một child, không chứng minh headroom container/cgroup. Checkpoint 59.422.875 bytes đã nạp lại trên CPU; fingerprint khớp GPU và đủ 64 updates. Phép thử inference dùng zero observation/all-silent mask, chưa phải mission rollout. Không có episode hoàn tất trong smoke này.

Review phát hiện fixture số nguyên cũ không bắt được quantization float16. Negative control gây thất bại thật ở validator cũ (DID NOT RAISE); đã thay bằng số lẻ float32, kiểm tra dtype và giá trị nguyên vẹn. Kiểm tra bổ sung đã COMPLETE trên cả hai Tesla T4 tại:
https://www.kaggle.com/code/vuliu123456/uav-corner-replay-float32-check-20261011
Cả hai thiết bị giữ nguyên roundtrip float32 và từ chối negative control ép float16. Đây là kiểm tra precision, không chạy train hoặc chứng minh hội tụ. Source manifests của các job được lưu trong JSON để phân biệt phiên bản code đã chạy với sửa đổi cuối.

## Trial số học ổn hơn vẫn cho hành vi thất bại

Trial 806d03ea đã có sẵn từ phiên trước:
https://www.kaggle.com/code/vuliu123456/uav-masac4096-b512-u32-trial1000-v1
Nó chỉ max_steps=1000, không phải paper 3000 steps. Kết thúc 4.096.000 transitions, 28.800 optimizer updates; critic loss 5,594, alpha_discrete 0,2270, alpha_continuous 0,001483, continuous entropy -27,4009. Dùng replay float16 cũ; không phải rollout của bản host float32 hiện tại. Các số cuối này chỉ cho biết checkpoint đó ít bất ổn số học hơn run gốc, không chứng minh training dài đã được sửa.

Đánh giá đúng checkpoint, config lịch sử và seed10044 trên CPU UavNetSim thật đã tạo đủ video và metrics:
https://www.kaggle.com/code/vuliu123456/uav-masac-trial1000-806d03ea-cpu-audit
Job bị ERROR khi bước cuối tìm source run trong project đã xóa; không phải lỗi tạo video. Các file media/manifest hoàn chỉnh đã được cứu và upload sang project mới, link eee3a6ce ở trên.

| Metric CPU trial | Giá trị |
|---|---:|
| Return | -153,865 |
| Coverage | 1,403% |
| Search / delivery | 0% / 0% |
| Target từng nằm trong FOV | 0 |
| APF intervention / boundary APF | 65,45% / 57,73% |
| Boundary clipping | 0,35% |
| Collision/separation violations | 17 trong 16 steps |
| Action mean x / y / z | -0,4530 / 0,1205 / -0,8126 |
| Action saturation | 85,97% |
| APF reward component | -130,9 |

Policy vẫn ra lệnh về nền/biên và không gặp mục tiêu. Tìm kiếm, xác nhận và delivery đều thất bại. CPU và tensor sensing có footprint khớp ở altitude 0, các độ cao dương nhỏ và tại biên; altitude 0 có bán kính sensing bằng 0 ở cả hai. Không tìm thấy bằng chứng từ probe này để sửa sensing/reward theo một lỗi parity giả định.

Evaluation độc lập trên CPU local bị timeout mà chưa ghi kết quả; không tính là pass hoặc dùng để kết luận hành vi. Các số ở bảng lấy từ manifest của lượt CPU Kaggle hoàn chỉnh, đã đối chiếu đúng SHA checkpoint.

## Giới hạn và phần chưa hoàn tất

Project mới, checkpoint, metrics còn lưu và media đã khôi phục. Các lỗi implementation đã tái hiện được đã sửa và có regression tests. Bản host float32 replay đã đi qua warmup, optimizer và save/restore trên hai GPU.

Chưa xác nhận policy sau fresh training với toàn bộ sửa đổi sẽ thoát góc hoặc đạt search/delivery. Reward APF đang phạt thật nhưng agent vẫn chọn hành vi kém; chưa đủ bằng chứng quy lỗi riêng cho hệ số reward hay kết luận phải thay APF. Phép thử tắt APF không cải thiện lệnh actor ở run gốc. Replay/warmup, temperature update và precision là những thay đổi có cơ sở kỹ thuật; hiệu quả dài hạn cần đánh giá bằng rollout sau train lại, cùng cảnh quan sát và nhiều seed.

Checkpoint cũ không tự học lại khi sửa mã hoặc tạo lại project W&B. Chưa chạy thêm lượt paper 20k trong đợt khôi phục này. Không công bố “đã sửa hết” cho chất lượng policy dựa vào loss hữu hạn, pass tests hoặc trạng thái finished của W&B.
