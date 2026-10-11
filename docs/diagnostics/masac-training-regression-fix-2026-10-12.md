# MASAC càng train càng tệ: bằng chứng và bản sửa actor encoder

Ngày kiểm tra: 12-10-2026 theo giờ Việt Nam. Repository `truongdaoanhduy/uav_search_network`, branch `fix/reward-safety-investigation`. Đây là chẩn đoán tiếp theo của `corner-followup-2026-10-12.md`; các con số mới đều là dữ liệu quan sát hoặc kiểm thử thật.

## Kết luận hiện tại

Baseline mới `03db0105` vẫn suy giảm hành vi sau những sửa replay/precision/temperature trước đó. Actor ra lệnh gần bão hòa về biên và độ cao thấp; APF liên tục sửa các lệnh này. Loss critic nhỏ không đồng nghĩa policy giải quyết nhiệm vụ tốt.

Đã xác nhận và sửa một lỗi kiến trúc: tầng feature cuối của actor bị xây như tầng output, thiếu activation và dùng gain khởi tạo linear. Với H512×256, actor chỉ có một tầng phi tuyến trước các head, trong khi critic có hai. Bản sửa bổ sung activation và gain tương ứng cho tầng 256, giữ tùy chọn LayerNorm theo cấu hình. Lỗi này có thể làm giảm khả năng biểu diễn policy; **chưa chứng minh nó là nguyên nhân duy nhất của suy giảm theo thời gian**.

Chưa có cơ sở thực nghiệm để thay APF hoặc tăng hệ số reward tùy ý. Phép thử chuyển động trước đây cho thấy tắt APF vẫn đưa lệnh actor vào góc; giữ APF với lệnh hướng vào trong vẫn thoát được. Đó là phép thử chuyển động giới hạn, không bảo đảm APF đúng/an toàn ở mọi tình huống.

## Metrics và hành vi thực tế

History baseline được scan đầy đủ trong lần kiểm tra trước: 12.288 bản ghi episode tại thời điểm scan. So sánh 256 episode đầu và 256 cuối:

| Metric | 256 đầu | 256 cuối |
|---|---:|---:|
| Return | -58,78 | -274,69 |
| Coverage | 52,58% | 21,24% |
| Search | 10,25% | 0,77% |
| Delivery | 1,30% | 0,25% |
| APF intervention | 28,97% | 64,49% |
| Boundary APF | 22,93% | 52,06% |
| Coverage redundancy | 96,46% | 98,80% |

Đây là các cohort training khác nhau của một seed, không phải đánh giá checkpoint trên cùng cảnh và không phải confidence interval qua nhiều seed.

Snapshot W&B mới nhất trong JSON đi kèm ghi baseline ở transition 27.418.624: critic loss 0,1081; alpha_continuous 0,0003593; saturation 93,71%. 141 episode trong 500 history rows cuối có coverage trung bình 12,85%, return -276,79 và APF 63,60%. Cửa sổ này không tương đương cửa sổ 256 ở bảng trên. B512 runs còn ở tiến độ khác nhau; không chọn winner từ các cửa sổ không khớp.

### Đánh giá CPU đủ episode của checkpoint 10.098

- Training run: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/03db0105
- CPU audit có video/PNG mới: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/c4e9e572
- Kaggle CPU-only: https://www.kaggle.com/code/bduy2378/uav-masac-baseline-10098-cpu-audit-20261012
- Checkpoint: `masac_episode_00010098.pt`, SHA-256 `0f05a3f0fc1cda76ec5012ae95b5ec7f6f0d9e423079e89b35d690a72d90c737`.
- Source chạy: `7e9b2020a86aaa2e500838c2ce8a8038c9e9685d`, seed evaluation 10044, horizon 3000, CPU UavNetSim thật, không sửa lệnh policy, reward hoặc APF.
- Job COMPLETE; W&B audit finished và run_role=checkpoint_audit. Đây là checkpoint định kỳ, không phải checkpoint final 20k.

| Metric CPU | Giá trị |
|---|---:|
| Return | -337,635 |
| Coverage | 0,8403% |
| Search / delivery | 0% / 0% |
| Target từng nằm trong FOV | 0 |
| Episode length | 1919 |
| APF intervention / boundary APF | 73,21% / 53,48% |
| Saturation | 99,10% |
| Motion mean x / y / z | +0,5679 / +0,6373 / -0,4787 |
| UAV cạn pin | 6/6 |
| APF reward component | -280,967 |

Cả 6 UAV kết thúc cách biên ngang gần nhất khoảng 3,97–7,10 m, tập trung ở góc x≈3000, y≈3000. Trace lưu mỗi 10 steps: 54,60% điểm độ cao dưới 10 m. Đây là tỷ lệ trên các điểm trace được lấy mẫu, không phải tích phân chính xác mọi bước. PNG gốc được lưu ở `checkpoint-10098-cpu-2026-10-12.png`; hash khớp media W&B.

`collision_violation_count=209` là số vi phạm khoảng cách/mô hình trong 108 steps, không phải 209 tai nạn vật lý độc lập. Silent 100% ở đây đi cùng 0 confirmation và 0 network request; không thể quy riêng thành lỗi truyền thông vì chưa có report để gửi.

Checkpoint 15.106 có hướng z dương tại trạng thái đầu, nhưng phép thử 24 bước vẫn đưa 5/6 UAV sát biên y=0. Do đó không suy rộng rằng mọi checkpoint luôn hướng xuống, và ép z dương không phải bản sửa đã triển khai.

## Vì sao loss nhỏ mà hành vi xấu

Critic loss đo khả năng khớp Bellman target trên minibatch của replay hiện tại; nó không đo coverage, confirmation hay delivery. Khi policy lặp hành động ở biên, dữ liệu mới cũng tập trung ở những trạng thái đó. Critic có thể khớp tốt target của quỹ đạo kém trong khi actor thiếu gradient có ích để tìm hành động tốt hơn.

Đây là diễn giải phù hợp với loss nhỏ, saturation cao, redundancy cao và mission metrics thấp; chưa phải chứng minh nhân quả đầy đủ. Các rủi ro cần ablation:

1. FIFO 409.600 transitions với 4096 env chỉ giữ khoảng 100 vector steps, trong khi episode dài 1300–3000 steps. Guard 100 steps đã sửa buffer quá ngắn trước đây nhưng không bảo đảm replay giữ đầy đủ kinh nghiệm mission. Thay bằng reservoir/stratification hoặc giảm env width là quyết định thiết kế cần đối chứng, chưa triển khai tùy ý.
2. Alpha_continuous rất nhỏ đi cùng saturation cao mô tả thiếu khám phá ở trạng thái quan sát, nhưng không tự chứng minh công thức temperature sai. Dấu và target âm theo action dimension hiện dùng khớp quy ước SAC thông thường: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/sac/sac.html .
3. Learner đang cộng entropy continuous cho toàn bộ UAV. Chưa xác định đây là nguyên nhân của giai đoạn suy giảm khi tất cả còn active; thay active/action mask sẽ đổi objective và cần phép thử riêng. Không ghi giả thuyết này thành lỗi đã sửa.
4. Reward APF đang phạt thật và là thành phần âm lớn của rollout. Chưa có chứng cứ rằng chỉ tăng phạt sẽ khiến policy thoát góc thay vì tìm một cực tiểu khác.

## Sửa code và tương thích checkpoint

`uav_marl/algorithms/masac.py`:

- `build_actor_encoder` phiên bản 2 kích hoạt tầng hidden cuối, thêm LayerNorm khi cấu hình yêu cầu, khởi tạo gain đúng theo activation.
- Actor/trainer lưu `actor_encoder_version` trong trainer_state.
- `load_masac_checkpoint` đọc marker trước khi dựng actor/optimizer; checkpoint lịch sử thiếu marker mặc định phiên bản 1.
- Direct trainer state loading từ chối mismatch trước khi sửa weights, tránh âm thầm diễn giải weights cũ bằng mạng mới.

`uav_marl/algorithms/matd3.py` dùng chung builder, nên được cập nhật versioning tương ứng, bao gồm target actor. Không thay stochastic MASAC, twin critics, CTDE, entropy loss, gamma/tau, mission scenario hay APF/reward.

Checkpoint cũ không tự học lại khi sửa source. Nạp lại bằng full checkpoint loader giữ chính xác kiến trúc cũ và hành động cũ. Raw actor-only `nn.Module.load_state_dict` không có guard này; production callers hiện dùng full trainer/checkpoint loaders. Không chuyển raw weights lịch sử trực tiếp vào actor mới để đánh giá.

## Verification

- Regression mới trước sửa: **8 failed, 4 passed**. Thất bại đúng ở thiếu activation, thiếu marker và không từ chối direct mismatch.
- Regression gain tầng hidden cuối trước sửa: variance 0,00097705, cần 0,00195293, test failed.
- Sau sửa: **13 focused tests passed**.
- Test checkpoint legacy kiểm tra outputs bit-for-bit, Adam momentum/step/parameter ordering; bao gồm LeakyReLU NoLN và PReLU với LayerNorm, cả MASAC/MATD3 và target actor.
- Full suite: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RUN_UAVNETSIM_INTEGRATION=1 python -m pytest`: **264 passed, 0 failed, 0 skipped, 416,50 s**.
- Compileall và git diff --check thành công.
- Review độc lập `encoder_review`: không có Critical/Important findings; xác nhận CPU/vector resume dùng loader version-aware.

Các kiểm thử trên xác nhận sửa kiến trúc và tính tương thích, không chứng minh policy đã hội tụ hoặc thoát góc sau fresh training.

## Đối chứng tiếp theo và giới hạn

Chuẩn bị một fresh run riêng với actor v2, cùng baseline B256/effective U16, 4096 env, seed44, replay/warmup409600 float32 CPU, cùng reward/APF và mission horizon3000, target20k. Source được pin vào commit đã kiểm thử. Final checkpoint phải được đánh giá CPU seed10044 và có video/PNG trước khi kết luận cải thiện.

Đối chứng này là một seed và thay activation cùng gain khởi tạo; việc dựng mạng mới cũng thay chuỗi số ngẫu nhiên khởi tạo. Không gọi chênh lệch kết quả một seed là bằng chứng nhân quả thống kê. Năm production runs trước đó vẫn giữ source `7e9b202` và cấu hình ban đầu; không restart chúng.

Tình trạng launch và ID job/run cụ thể sẽ được ghi trong `actor-encoder-v2-ablation-2026-10-12.json`. Nếu chưa có manifest submitted/running, không tính run mới là đã chạy. Kết quả chất lượng của bản sửa vẫn chưa có.
