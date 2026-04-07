from mesa import Agent
import random
from modules.psychology import PsychologyModel

# ─────────────────────────────────────────────────────────────────────────────
# Ajan tiplerine göre başlangıç parametreleri
# ─────────────────────────────────────────────────────────────────────────────
AGENT_TYPE_PROFILES = {
    "ideal": {
        "base_trust":         65,
        "autonomy":           1.0,
        "satisfaction":       0.7,
        "max_usage_duration": 5,
        # w1=Trust, w2=Satisfaction, w3=Autonomy, w4=Scarcity
        "w1_range": (0.40, 0.50),
        "w2_range": (0.30, 0.45),
        "w3_range": (0.15, 0.25),
        "w4_range": (0.05, 0.15),
    },
    "standard": {
        "base_trust":         50,
        "autonomy":           0.7,
        "satisfaction":       0.5,
        "max_usage_duration": 5,
        "w1_range": (0.30, 0.50),
        "w2_range": (0.30, 0.50),
        "w3_range": (0.10, 0.30),
        "w4_range": (0.10, 0.30),
    },
    "toxic": {
        "base_trust":         30,
        "autonomy":           0.4,
        "satisfaction":       0.3,
        "max_usage_duration": 10,   # doğal olarak kaynağı daha uzun tutar
        "w1_range": (0.15, 0.30),
        "w2_range": (0.20, 0.35),
        "w3_range": (0.10, 0.20),
        "w4_range": (0.25, 0.35),   # kıtlığa çok duyarlı → rekabetçi
    },
}

# Kaç adım arka arkaya kaynak bulunamazsa "hayal kırıklığı" tetiklenir
FRUSTRATION_THRESHOLD = 3


class PersonAgent(Agent):
    def __init__(self, model, agent_type: str = "standard"):
        super().__init__(model)

        self.agent_type = agent_type
        profile = AGENT_TYPE_PROFILES.get(agent_type, AGENT_TYPE_PROFILES["standard"])

        # ── Kaynak kullanım durumu ──────────────────────────────────────────
        self.current_resource = None
        self.usage_duration = 0
        self.wait_time = 0
        self.is_defecting = False
        self.max_usage_duration = profile["max_usage_duration"]

        # ── Kümülatif kullanım (Gini katsayısı hesabı için) ────────────────
        self.cumulative_usage = 0

        # ── Hayal kırıklığı sayacı ─────────────────────────────────────────
        self.frustration_counter = 0

        # ── Psikolojik durum değişkenleri ──────────────────────────────────
        # Güven: model'in sistem tipine göre AI penaltısı uygulanır
        is_ai = model.system_type in ("ai_advisory", "ai_autonomous", "integrated")
        self.trust = PsychologyModel.calculate_initial_trust(
            base_trust=profile["base_trust"],
            is_ai_system=is_ai,
        )
        self.autonomy = profile["autonomy"]
        self.satisfaction = profile["satisfaction"]

        # Komşu gözleminden hesaplanan topluluk adalet algısı (0-1)
        self.perceived_community_fairness = 0.5

        # ── İşbirliği karar ağırlıkları (tipe özel aralıklardan rastgele) ──
        self.weights = (
            random.uniform(*profile["w1_range"]),
            random.uniform(*profile["w2_range"]),
            random.uniform(*profile["w3_range"]),
            random.uniform(*profile["w4_range"]),
        )
        self.last_iforest_label = 1
        self.last_sanction_tick = -1

    # ─────────────────────────────────────────────────────────────────────────
    # Ana adım döngüsü
    # ─────────────────────────────────────────────────────────────────────────

    def step(self):
        """
        Ana karar döngüsü:
        - Kaynak tutmuyorsa: bekle, algıla, karar ver, talep et.
        - Kaynak tutuyorsa: kullan, süresi dolunca bırak.
        """
        if self.current_resource is None:
            self.wait_time += 1
            self.request_resource()
        else:
            self.use_resource()

    # ─────────────────────────────────────────────────────────────────────────
    # Kaynak talep etme
    # ─────────────────────────────────────────────────────────────────────────

    def request_resource(self):
        if not self.model.can_agent_request(self):
            self.frustration_counter += 1
            self.autonomy = PsychologyModel.update_autonomy(
                self.autonomy, "forced_decision"
            )
            return

        free_resources = self.model.get_free_resources()

        # Kıtlık algısı: ne kadar az boş kaynak varsa o kadar yüksek
        total_resources = getattr(self.model, "num_resources", 10)
        scarcity = 1.0 - (len(free_resources) / max(1, total_resources))

        # İşbirliğinin öznel maliyeti: yüksek özerklik maliyeti düşürür
        base_cost = 1.0
        effective_cost = PsychologyModel.calculate_cooperation_cost(
            base_cost=base_cost,
            autonomy_felt=self.autonomy,
        )
        # Efektif maliyet düşükse işbirliği daha kolay → eşiği yumuşat
        cost_factor = 1.0 - min(0.3, (base_cost - effective_cost))

        # İşbirliği olasılığı (maliyet faktörüyle ölçeklendirilmiş)
        p_coop = PsychologyModel.calculate_cooperation_probability(
            self.trust,
            self.satisfaction,
            self.autonomy,
            scarcity,
            self.weights,
        ) * cost_factor

        p_coop = self.model.adjust_cooperation_probability(self, p_coop)
        self.is_defecting = random.random() >= p_coop

        if free_resources:
            # Boş kaynak bulundu → anlamlı seçim hakkı kullanıldı
            resource = random.choice(free_resources)
            resource.is_occupied = True
            resource.user = self
            self.current_resource = resource
            self.usage_duration = 0

            # Özerklik: anlamlı bir seçim yapıldı
            self.autonomy = PsychologyModel.update_autonomy(
                self.autonomy, "meaningful_choice"
            )
            # Başarılı erişim → hayal kırıklığı sıfırlanır
            self.frustration_counter = 0

            if getattr(self.model, "verbose", False):
                print(
                    f"Agent {self.unique_id} ({self.agent_type}) acquired: "
                    f"{resource.unique_id}"
                )
            self.model.on_resource_acquired(self)
        else:
            # Kaynak yok → hayal kırıklığı sayacı artar
            self.frustration_counter += 1

            if self.frustration_counter >= FRUSTRATION_THRESHOLD:
                # Sistem kaynak sağlayamıyor: güven ve özerklik düşer
                self.trust = PsychologyModel.update_trust(self.trust, "negative")
                self.autonomy = PsychologyModel.update_autonomy(
                    self.autonomy, "forced_decision"
                )
                self.frustration_counter = 0
            self.model.on_resource_unavailable(self)

    # ─────────────────────────────────────────────────────────────────────────
    # Kaynak kullanımı
    # ─────────────────────────────────────────────────────────────────────────

    def use_resource(self):
        """
        Kullanım süresini yönetir. Defector ajanlar kaynağı daha uzun tutar.
        """
        self.usage_duration += 1

        limit = self.max_usage_duration
        if self.is_defecting:
            limit = limit * 2

        if self.usage_duration >= limit:
            self.release_resource()

    # ─────────────────────────────────────────────────────────────────────────
    # Kaynağı bırakma + geri bildirim döngüsü
    # ─────────────────────────────────────────────────────────────────────────

    def release_resource(self):
        """
        Kaynağı serbest bırakır ve psikolojik güncelleme döngüsünü tetikler:
        güven, memnuniyet (DEA + eşitlik bonusu + prosedürel adalet) ve özerklik.
        """
        if not self.current_resource:
            return

        if getattr(self.model, "verbose", False):
            print(
                f"Agent {self.unique_id} ({self.agent_type}) released: "
                f"{self.current_resource.unique_id}"
            )
        self.current_resource.is_occupied = False
        self.current_resource.user = None
        self.current_resource = None
        self.model.on_resource_released(self)

        # ── Komşu gözlemi ve eşitlik algısı ──────────────────────────────
        community_avg_wait = self.observe_neighbors()

        # Eşitlik bonusu: kendi bekleme süresini topluluk ortalamasıyla kıyasla.
        # Ortalamanın altındaysak adaletli hissediyoruz → pozitif bonus
        # Ortalamanın üstündeyse haksızlık algısı → negatif bonus
        if community_avg_wait > 0:
            equity_ratio = self.wait_time / community_avg_wait
            # equity_ratio < 1 → kısa bekledik (iyi), > 1 → uzun bekledik (kötü)
            equity_bonus = max(-0.2, min(0.2, (1.0 - equity_ratio) * 0.2))
        else:
            equity_bonus = 0.0

        # Prosedürel adalet: modelden gelen sistem bonusu + eşitlik gözlemi
        procedural_bonus = equity_bonus + self.model.procedural_bonus_modifier

        # ── DEA-benzeri memnuniyet hesabı ─────────────────────────────────
        inputs = self.wait_time + 1.0
        outputs = self.usage_duration

        new_sat = PsychologyModel.calculate_satisfaction(
            inputs_x=inputs,
            outputs_y=outputs,
            weights_v=1.0,
            weights_u=1.0,
            procedural_bonus=procedural_bonus,
        )

        # Ağırlıklı hareketli ortalama ile yumuşatılmış güncelleme
        self.satisfaction = (self.satisfaction * 0.7) + (new_sat * 0.3)

        # ── Güven: başarılı kullanım → pozitif deneyim ────────────────────
        self.trust = PsychologyModel.update_trust(self.trust, "positive")

        # ── Kümülatif kullanım kaydı (Gini için) ─────────────────────────
        self.cumulative_usage += self.usage_duration

        self.wait_time = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Komşu gözlemi (Eşitlik Teorisi için)
    # ─────────────────────────────────────────────────────────────────────────

    def observe_neighbors(self) -> float:
        """
        Grid'deki komşu PersonAgent'larının wait_time ortalamasını hesaplar.
        Sonucu perceived_community_fairness olarak saklar ve döndürür.

        Returns:
            float: Komşuların ortalama wait_time değeri (komşu yoksa 0).
        """
        if not hasattr(self.model, "grid") or self.pos is None:
            return 0.0

        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=2
        )

        neighbor_wait_times = [
            n.wait_time
            for n in neighbors
            if isinstance(n, PersonAgent)
        ]

        if not neighbor_wait_times:
            return 0.0

        avg_wait = sum(neighbor_wait_times) / len(neighbor_wait_times)

        # 0-1 normalizasyonu: uzun bekleme = düşük adalet algısı
        # 10 adım beklemeyi "kötü" için referans kabul ediyoruz
        self.perceived_community_fairness = max(0.0, 1.0 - (avg_wait / 10.0))

        return avg_wait
