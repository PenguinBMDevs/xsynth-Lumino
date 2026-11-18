use crate::soundfont::VoiceSpawner;

use crate::voice::{Voice, VoiceControlData};

pub struct VoiceSpawnerMatrix {
    voice_spawners_attack: Vec<Vec<Box<dyn VoiceSpawner>>>,
    voice_spawners_release: Vec<Vec<Box<dyn VoiceSpawner>>>,
    /// 构造批 voice（仅单线程批渲染启用；并行 key 渲染下批 voice 只能标量回退）。
    batch_voices: bool,
}

fn voice_iter_from_vec<'a>(
    vec: &'a [Box<dyn VoiceSpawner>],
    control: &'a VoiceControlData,
) -> impl Iterator<Item = Box<dyn Voice>> + 'a {
    vec.iter().map(move |voice| voice.spawn_voice(control))
}

fn exclusive_classes_from_vec<'a>(
    vec: &'a [Box<dyn VoiceSpawner>],
) -> impl Iterator<Item = u8> + 'a {
    vec.iter().filter_map(|voice| voice.exclusive_class())
}

impl VoiceSpawnerMatrix {
    pub fn new() -> Self {
        let mut voice_spawners_attack = Vec::new();
        let mut voice_spawners_release = Vec::new();

        for _ in 0..(128 * 128) {
            voice_spawners_attack.push(Vec::new());
            voice_spawners_release.push(Vec::new());
        }

        voice_spawners_attack.shrink_to_fit();
        voice_spawners_release.shrink_to_fit();

        VoiceSpawnerMatrix {
            voice_spawners_attack,
            voice_spawners_release,
            batch_voices: true,
        }
    }

    /// 设置是否构造批 voice，并立即作用于已注册的 spawner（配置可在 soundfont 之后变更）。
    pub fn set_batch_voices(&mut self, enabled: bool) {
        if self.batch_voices == enabled {
            return;
        }
        self.batch_voices = enabled;
        for vec in self
            .voice_spawners_attack
            .iter_mut()
            .chain(self.voice_spawners_release.iter_mut())
        {
            for spawner in vec.iter_mut() {
                spawner.set_batch_voices(enabled);
            }
        }
    }

    #[inline(always)]
    fn get_spawners_index_at_attack(&self, key: u8, vel: u8) -> usize {
        key as usize + vel as usize * 128
    }

    #[inline(always)]
    fn get_spawners_index_at_release(&self, key: u8, vel: u8) -> usize {
        key as usize + vel as usize * 128
    }

    #[inline(always)]
    fn get_attack_spawners_vec_at(&self, key: u8, vel: u8) -> &Vec<Box<dyn VoiceSpawner>> {
        &self.voice_spawners_attack[self.get_spawners_index_at_attack(key, vel)]
    }

    #[inline(always)]
    fn get_release_spawners_vec_at(&self, key: u8, vel: u8) -> &Vec<Box<dyn VoiceSpawner>> {
        &self.voice_spawners_release[self.get_spawners_index_at_release(key, vel)]
    }

    #[inline(always)]
    pub fn spawn_voices_attack<'a>(
        &'a self,
        control: &'a VoiceControlData,
        key: u8,
        vel: u8,
    ) -> impl Iterator<Item = Box<dyn Voice>> + 'a {
        voice_iter_from_vec(self.get_attack_spawners_vec_at(key, vel), control)
    }

    #[inline(always)]
    pub fn spawn_voices_release<'a>(
        &'a self,
        control: &'a VoiceControlData,
        key: u8,
        vel: u8,
    ) -> impl Iterator<Item = Box<dyn Voice>> + 'a {
        voice_iter_from_vec(self.get_release_spawners_vec_at(key, vel), control)
    }

    #[inline(always)]
    pub fn set_spawners_attack(
        &mut self,
        key: u8,
        vel: u8,
        mut spawners: Vec<Box<dyn VoiceSpawner>>,
    ) {
        if !self.batch_voices {
            for spawner in spawners.iter_mut() {
                spawner.set_batch_voices(false);
            }
        }
        let index = self.get_spawners_index_at_attack(key, vel);
        self.voice_spawners_attack[index] = spawners;
    }

    #[inline(always)]
    pub fn set_spawners_release(
        &mut self,
        key: u8,
        vel: u8,
        mut spawners: Vec<Box<dyn VoiceSpawner>>,
    ) {
        if !self.batch_voices {
            for spawner in spawners.iter_mut() {
                spawner.set_batch_voices(false);
            }
        }
        let index = self.get_spawners_index_at_release(key, vel);
        self.voice_spawners_release[index] = spawners;
    }

    #[inline(always)]
    pub fn exclusive_classes_attack<'a>(
        &'a self,
        key: u8,
        vel: u8,
    ) -> impl Iterator<Item = u8> + 'a {
        exclusive_classes_from_vec(self.get_attack_spawners_vec_at(key, vel))
    }
}
